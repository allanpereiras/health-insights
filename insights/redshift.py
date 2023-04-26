from __future__ import annotations

import contextlib  # Fix Type Hinting for model Classes
from datetime import date, datetime
from enum import Enum, unique

import pandas as pd
import pytz
from django.conf import settings
from django.db import IntegrityError, connections, transaction
from django.db.models import F, Q
from django.db.utils import OperationalError
from django.utils.dateparse import parse_date, parse_datetime
from django.utils.timezone import is_aware, make_aware, now, utc
from insights import models
from patients.utils import get_or_create_patient_with_external_id as get_patient


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Redshift(metaclass=Singleton):
    """Class for handling Redshift connection and operations"""

    DB_NAME = "redshift"

    @staticmethod
    def parse_dt(dt):
        """Parses a datetime object to the one used in Redshift"""
        return dt.strftime("%Y-%m-%d %H:%M:%S%z")

    def dictfetchall(self, cursor):
        """Return all rows from a cursor as a dict"""
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def dictfetchone(self, cursor):
        rows = self.dictfetchall(cursor)
        if rows:
            return rows[0]
        return rows

    def get_cursor(self):
        """Returns connection cursor for querying in Redshift"""
        return connections[self.DB_NAME].cursor()

    def execute(self, query, params, many=True, as_dict=True):
        """Executes cursor with provided arguments for query and params.
        Raises OperationalError if querying Redshift
        """
        if settings.TESTING:
            from insights.tests.factories import RedshiftConnection
            return RedshiftConnection().retrieve_data()
        with self.get_cursor() as c:
            try:
                c.execute(query, params)
                if as_dict:
                    return self.dictfetchall(c) if many else self.dictfetchone(c)
                return c.fetchall() if many else c.fetchone()
            except Exception as e:
                raise OperationalError(f"Redshift query error: {e}\n{c.query}") from e

    def filter(self, ext_patient_id, dt_from, dt_to, d_marker=None, as_dict=True):
        """Filter rows using provided arguments. Distributor Marker is optional
        :param ext_patient_id:  External Patient id
        :type ext_patient_id:   int
        :param dt_from:         Start date time in UTC
        :type dt_from:          datetime aware
        :param dt_to:           End date time in UTC
        :type dt_to:            datetime aware
        :param d_marker:        Distributor Marker name.
        :type d_marker:         str
        :return:                Dictionary of value, value_timestamp and
                                distributor_marker if not provided.
        :rtype:                 dict
        """
        if not d_marker:
            query = (
                "SELECT distributor_marker, value, value_timestamp "
                "FROM health_standardized WHERE external_patient_id = %s "
                "AND value_timestamp BETWEEN %s AND %s"
            )  # Do NOT quote %s -> SQL Injection risk
            params = [ext_patient_id, self.parse_dt(dt_from), self.parse_dt(dt_to)]
        else:
            query = (
                "SELECT value, value_timestamp FROM health_standardized "
                "WHERE external_patient_id = %s AND distributor_marker = %s AND "
                "value_timestamp BETWEEN %s AND %s"
            )  # Do NOT quote %s -> SQL Injection risk
            params = [
                ext_patient_id,
                d_marker,
                self.parse_dt(dt_from),
                self.parse_dt(dt_to),
            ]
        return self.execute(query, params, as_dict=as_dict)

    def latest(self, d_marker, ext_patient_id=None, as_dict=True):
        """Get latest row data for a Distributor Marker.
        :param d_marker:        Distributor Marker name.
        :type d_marker:         str
        :param ext_patient_id:  External Patient id (Optional)
        :type ext_patient_id:   int
        :return:                Dictionary of value, value_timestamp
        :rtype:                 dict
        """
        if not ext_patient_id:
            query = (
                "SELECT h.value, h.value_timestamp FROM health_standardized h WHERE "
                "distributor_marker = %s AND h.value_timestamp = ("
                "SELECT MAX(h2.value_timestamp) FROM health_standardized h2)"
            )  # Do NOT quote %s -> SQL Injection risk
            params = [d_marker]
        else:
            query = (
                "SELECT h.value, h.value_timestamp FROM health_standardized h WHERE "
                "external_patient_id = %s AND distributor_marker = %s AND "
                "h.value_timestamp = (SELECT MAX(h2.value_timestamp) from "
                "health_standardized h2)"
            )  # Do NOT quote %s -> SQL Injection risk
            params = [ext_patient_id, d_marker]
        return self.execute(query, params, as_dict=as_dict)


class BaseCalculation:
    def __init__(self, dt_from: datetime, dt_to: datetime, ext_patient_id: int = None):
        self._patient_id = None
        self.ext_patient_id = ext_patient_id
        if not isinstance(dt_from, datetime):
            dt_from = parse_datetime(dt_from)
        if not isinstance(dt_to, datetime):
            dt_to = parse_datetime(dt_to)
        self.dt_from = dt_from if is_aware(dt_from) else make_aware(dt_from)  # Timezone
        self.dt_from = self.dt_from.astimezone(tz=utc)  # Force UTC
        self.dt_to = dt_to if is_aware(dt_to) else make_aware(dt_to)  # Timezone
        self.dt_to = self.dt_to.astimezone(tz=utc)  # Force UTC
        self.redshift = Redshift()

    @property
    def patient_id(self):
        if self._patient_id is None and self.ext_patient_id is not None:
            self._patient_id = get_patient(self.ext_patient_id)[0].id
        return self._patient_id


class DistribMarkerStatistics(BaseCalculation):
    """Defines a Distributor Marker Statistics to be calculated in Redshift.
    The Statistics are tied to a date time Period and can be applied
    for each Distributor Marker of all Patients or of a single Patient.

    Attributes:
        distrib_marker: Distributor Marker name
        dt_from:        Start date time in UTC
        dt_to:          End date time in UTC
        ext_patient_id: External Patient id (None for all Patients)
        patient_id:     Patient Id (None for all Patients)
        redshift:       Redshift instance connection
    """

    @unique
    class AggregateFunctions(Enum):
        AVG = "AVG(value)"
        MIN = "MIN(value)"
        MAX = "MAX(value)"
        SUM = "SUM(value)"

        @classmethod
        def is_valid(cls, agg_func: str) -> bool:
            return agg_func in cls.__members__

        @classmethod
        def get_agg_func(cls, agg_func: str) -> str:
            """Verifies if Aggregating Function provided as argument is valid.
            :param agg_func:    Available aggregate function.
            :type agg_func:     str
            :raise KeyError:    If invalid Aggregate Function
            """
            try:  # Protects SQL Injection
                return cls[agg_func.upper()].value
            except KeyError as e:
                raise ValueError(f"Undefined Aggregate Function {agg_func}") from e

    def __init__(self, distrib_marker: str, *args, **kwargs):
        # External Patient ID is optional for calculations from BaseCalculation
        self.distrib_marker = distrib_marker
        self._device_name = kwargs.pop("device_name", None)  # Optional for filtering
        super().__init__(*args, **kwargs)
        self.__dict__.update((f"_{agg.name}", None) for agg in self.AggregateFunctions)
        self._latest = None

    @property
    def device_name(self):
        return self._device_name

    @device_name.setter
    def device_name(self, value: str) -> None:
        if not isinstance(value, str) and value is not None:
            raise ValueError("Device name needs to be a string")
        self._device_name = value

    @property
    def AVG(self):
        if self._AVG is None:
            setattr(self, "AVG", self.calculate(self.AggregateFunctions.AVG.name))
        return self._AVG

    @AVG.setter
    def AVG(self, value):
        self._AVG = value

    @property
    def MIN(self):
        if self._MIN is None:
            setattr(self, "MIN", self.calculate(self.AggregateFunctions.MIN.name))
        return self._MIN

    @MIN.setter
    def MIN(self, value):
        self._MIN = value

    @property
    def MAX(self):
        if self._MAX is None:
            setattr(self, "MAX", self.calculate(self.AggregateFunctions.MAX.name))
        return self._MAX

    @MAX.setter
    def MAX(self, value):
        self._MAX = value

    @property
    def SUM(self):
        if self._SUM is None:
            setattr(self, "SUM", self.calculate(self.AggregateFunctions.SUM.name))
        return self._SUM

    @SUM.setter
    def SUM(self, value):
        self._SUM = value

    def get_latest(self, force_refresh: bool = False) -> dict:
        """Returns latest row for this Distributor Marker and Patient"""
        if self._latest is None or force_refresh:
            self._latest = self.redshift.latest(
                self.distrib_marker, ext_patient_id=self.ext_patient_id
            )
        return self._latest

    def _build_query(self, agg_func) -> tuple:
        agg_func = self.AggregateFunctions.get_agg_func(agg_func)  # Protects SQLInject
        agg_func = agg_func.replace("value", "t1.value")
        columns = [
            "t1.distributor_marker",
            agg_func,
            "t1.value_units",
            "max(t1.value_timestamp)",
        ]
        constrs = ["distributor_marker"]
        params = [self.distrib_marker]
        group_by = ["distributor_marker", "value_units"]
        if self.ext_patient_id or self.device_name:
            columns.insert(0, "t1.device_name")
            group_by.insert(0, "device_name")
        if self.device_name:
            constrs.insert(0, "device_name")
            params.insert(0, self.device_name)
        if self.ext_patient_id:
            constrs.insert(0, "external_patient_id")
            params.insert(0, self.ext_patient_id)
        constrs = [
            f"h.{c} = %s" for c in constrs
        ]  # Do NOT quote %s -> SQL Injection risk
        where = f"WHERE {' AND '.join(constrs)}"
        where += " AND h.value_timestamp_start >= %s AND h.value_timestamp_end <= %s"
        # Clear duplicates rows with distinct on value_timestamp_start (start dt)
        # Most recent value is the one we want for each distribution marker and start dt
        no_dup = (
            "WITH no_dup as ("
            "   SELECT h.*, ROW_NUMBER() OVER (PARTITION BY distributor_marker, "
            "   value_timestamp_start ORDER BY created_at DESC) as rn "
            f"  FROM health_standardized h {where}) "
        )
        select = (
            f"SELECT {', '.join(columns)} "
            "FROM no_dup t1 "
            "WHERE t1.rn=1 AND NOT EXISTS("
            "    SELECT * FROM no_dup t2 WHERE "
            "       t1.value_timestamp_start > t2.value_timestamp_start "
            "   AND t1.value_timestamp_start < t2.value_timestamp_end "
            "   AND t1.device_name = t2.device_name "  # Allow different devices to overlap
            "   AND t1.value_timestamp_start != t2.value_timestamp_start "
            "   AND t1.value_timestamp_end != t2.value_timestamp_end )"
        )  # Select all rows from no_dup that do not have interval overlapping
        group_by = [f"t1.{c}" for c in group_by]
        group_by = f"GROUP BY {', '.join(group_by)}"
        query = " ".join([select, group_by])
        query = f"{no_dup} {query}"

        params += [
            self.redshift.parse_dt(self.dt_from),
            self.redshift.parse_dt(self.dt_to),
        ]
        return query, params

    def calculate(self, agg_func: str) -> dict:
        """Runs a SELECT aggregate based on instance attributes.
        :param agg_func:   Available aggregate function.
        :type agg_func:    str
        :raise ValueError: If invalid aggregate function.
        :return:           Dictionary with calculated value for the applied
                           aggregate function and its timestamp calculation
        :rtype:            dict
        """
        query, params = self._build_query(agg_func)
        return self.redshift.execute(query, params, many=True, as_dict=True)

    def refresh(self, agg_func: str) -> None:
        """Recalculates the Aggregating Function provided as argument
        :param agg_func:    Available aggregate function.
        :type agg_func:     str
        """
        setattr(self, agg_func, self.calculate(agg_func))

    def refresh_all(self) -> None:
        """Recalculates all Aggregating Functions for Patient Statistic"""
        # FIXME use single query with all Aggregating Functions (?)
        for agg in self.AggregateFunctions:
            setattr(self, agg.name, self.calculate(agg.name))


class DeviceAggregatedData(BaseCalculation):
    def __init__(self, device_name: str, *args, **kwargs):
        # External Patient ID is optional for calculations from BaseCalculation
        self.device_name = device_name
        super().__init__(*args, **kwargs)
        self.distributors_markers = set()

    def add_distributor_marker(self, d_marker: str):
        attrs = {
            "ext_patient_id": self.ext_patient_id,
            "device_name": self.device_name,
            "distrib_marker": d_marker,
            "dt_from": self.dt_from,
            "dt_to": self.dt_to,
        }
        d_marker = DistribMarkerStatistics(**attrs)
        self.distributors_markers.add(d_marker)

    def find_value(self, func: function, d_marker: str, agg_func: str) -> dict:
        if not DistribMarkerStatistics.AggregateFunctions.is_valid(agg_func):
            raise ValueError(f"`{agg_func}` is not a valid Aggregate Function name")
        d_markers = list(  # Filter Distributor Marker in self.distributors_markers
            filter(lambda d: d.distrib_marker == d_marker, self.distributors_markers)
        )
        if d_markers:
            values = []
            for d_marker in d_markers:
                value = getattr(d_marker, agg_func)  # Calculates if attr not in obj
                if value:
                    values += value
            agg = agg_func.lower()
            return func(values, key=lambda result: result[agg]) if values else None
        return None


class PatientInsights:
    """List and updates all of a Patient's Health Insights from a given
        collection date.

    Attributes:
        ext_patient_id:         External Patient id
        patient_id:             Patient id
        collection_date:        Collection date
        timezone:               Timezone info
        health_insights:        Queryset of PatientHealhtInsight
        distributors_markers:   List of Distributors Markers of the Insights
        statistics:             The Statistics for each Distributor Marker
        redshift:               Redshift instance connection
        values:                 Dictionary for all of the Patient's Distributor
                                Marker calculated statistics and its timestamp
    """

    # Refreshable properties order matters -> data depends on each other
    REFRESHABLE = {"health_insights", "distributors_markers"}

    def __init__(self, ext_patient_id: int, collection: date, timezone: str):
        self.ext_patient_id = ext_patient_id
        self._patient = None
        self._health_insights = None
        self._distributors_markers = None
        self._data_distributors = None
        self.values = {}
        if not isinstance(collection, date):
            collection = parse_date(collection)
            if not collection:
                raise ValueError("Bad Collection date")
        self.collection_date = collection
        self.timezone = timezone
        self.redshift = Redshift()
        self._devices = None

    @property
    def patient(self):
        if self._patient is None:
            self._patient = get_patient(self.ext_patient_id)[0]
        return self._patient

    @property
    def collection_date(self):
        return self._collection_date

    @collection_date.setter
    def collection_date(self, value: date) -> None:
        if value > now().date():
            raise ValueError("Collection date cannot be after today")
        self._collection_date = value

    @property
    def devices(self):
        if self._devices is None:
            devices = [
                DeviceAggregatedData(f"{d[0]}", ext_patient_id=self.ext_patient_id)
                for d in self.fetch_devices()
            ]
            setattr(self, "devices", devices)
        return self._devices

    @devices.setter
    def devices(self, value):
        self._devices = value

    @property
    def timezone(self):
        return self._timezone

    @timezone.setter
    def timezone(self, value: date) -> None:
        if value not in pytz.all_timezones:
            raise ValueError(
                f"Invalid time zone. `{value}` is not an IANA time zone database name"
            )
        self._timezone = pytz.timezone(value)

    @property
    def data_distributors(self):
        if self._data_distributors is None:
            self._data_distributors = self.health_insights.values(
                name=F("health_insight__marker__distributors__distributor__name")
            ).distinct()
        return self._data_distributors

    @property
    def distributors_markers(self):
        if self._distributors_markers is None:
            self._distributors_markers = self.health_insights.values(
                name=F("health_insight__marker__distributors__name")
            ).distinct()
        return self._distributors_markers

    @property
    def health_insights(self):
        if self._health_insights is None:
            self._health_insights = models.PatientHealthInsight.objects.filter(
                patient=self.patient,
                collection_date=self.collection_date,
            )
        return self._health_insights

    def get_period_for_insight(self, health_insight: models.HealthInsight) -> tuple:
        return health_insight.get_period(self.collection_date, tz=self.timezone)

    def update_insight_value(self, insight: models.PatientHealthInsight, **kwargs):
        result = self.fetch_insight_value(insight, **kwargs)
        d_marker = insight.get_distrib_marker()
        if result and d_marker:
            self.values.update({d_marker.name: result})
            return True
        return False

    def fetch_devices(self) -> list:
        """Fetches for this Patient all their devices names
           :return: List of devices names (str)        
        """
        query = (
            "SELECT DISTINCT device_name FROM health_standardized "
            "WHERE external_patient_id = %s "
        )  # Do NOT quote %s -> SQL Injection risk
        params = [self.ext_patient_id]
        return self.redshift.execute(query, params, many=True, as_dict=True)

    def fetch_insight_value(self, obj: models.PatientHealthInsight, **kwargs) -> dict:
        option = kwargs.get("option", "python") # Default option is python
        process_options = {
            "python": self.process_with_python,
            "sql": self.process_with_sql,
        }
        fetch_value = process_options.get(option)
        if not fetch_value:
            raise ValueError("Invalid process option")
        return fetch_value(obj)

    def get_in_values(self, insight: models.PatientHealthInsight, **kwargs):
        """Finds the calculated value for a Patient Health Insight.
            Returns a dict if found or None if not found.
        :param insight:    Patient Health Insight to find.
        :param kwargs:     Keyword arguments for the function.
        :param func (kwarg):    Function to calculate the value.
        :param agg (kwarg):     Aggregation function to use.
        :return:                Calculated value for the Patient Health Insight.
        """
        args = ["func", "agg"]
        if any(arg not in kwargs for arg in args): 
            missing = list(set(args) - set(kwargs.keys())) # Missing arguments
            raise ValueError(f"Missing `{missing}` argument(s)")
        func = kwargs["func"]
        agg = kwargs["agg"]
        d_marker = insight.get_distrib_marker()
        d_marker = d_marker.name if d_marker else None
        if d_marker not in self.values:
            self.update_insight_value(insight, **kwargs)
        all_devices_result = self.values.get(d_marker, None)
        if all_devices_result:
            result = all_devices_result[0]  # TODO: Handle multiple devices
            values_list = [result[agg] for result in all_devices_result]
            result[agg] = func(values_list)
            return result
        return None

    @transaction.atomic
    def get_or_create_insight(self, health_insight, distributor, **kwargs) -> tuple:
        """Get or Create the Patient Health Insight for the collection date.
            Performs validation of the Health Insight and Data Distributor.
        :param h_insight_id:    Health Insight id
        :type h_insight_id:     int
        :param distrib_id:      Data Distributor id
        :type distrib_id:       int
        :return:                PatientHealthInsight object and True if created
        :rtype:                 tuple
        """
        params = {
            "distributor_id": distributor.id,
            "marker_id": health_insight.marker.id,
        }
        distrib_maker = models.DistributorMarker.objects.filter(**params).first()
        distributor = distrib_maker.distributor if distrib_maker else None
        try:
            obj, created = models.PatientHealthInsight.objects.update_or_create(
                health_insight=health_insight,
                patient=self.patient,
                collection_date=self.collection_date,
                defaults={"data_distributor": distributor, "timezone": self.timezone},
            )  # Get or create for collection date
            if created or obj.value is None or kwargs.get("force_refresh"):
                self.update_in_orm(insight_ids=[obj.health_insight_id], **kwargs)
            return obj, created
        except IntegrityError as e:
            raise RuntimeError("Values do not comply with ORM constraints") from e

    @transaction.atomic
    def get_or_create_all_insights(self, distrib_name: str, **kwargs) -> list:
        """Returns the Patient Insights for all of the Health Insights
            associated with this Distributor Markers
        :return:    List of tuples consisting of a PatientHealthInsight object
                    and its indication of persist
        :rtype:     list
        """
        distrib = models.DataDistributor.objects.get(name=distrib_name)
        distrib_insights = models.HealthInsight.objects.filter(
            Q(marker__distributors__distributor=distrib)
            | Q(marker__distributors__isnull=True)
        )
        insight_list = [
            self.get_or_create_insight(h_insight, distrib, force_refresh=True, **kwargs)
            for h_insight in distrib_insights
        ]
        # Create the Patient Insights that are derived from combinations
        pairs_kwargs = {
            "data_distributor": distrib,
            "timezone": self.timezone,
            "last_collection_time": now().time(),
        }
        models.DistributorMarkerPair.create_patient_insights(
            self.patient.id, self.collection_date, **pairs_kwargs
        )
        result = insight_list
        if 'sync_devices' in kwargs and kwargs['sync_devices']:
            devices = self.fetch_devices()
            names = [d["device_name"] for d in devices]
            models.DataDevice.objects.bulk_create(
                [models.DataDevice(name=n, version="") for n in names], 
                ignore_conflicts=True
            )
            devices = models.DataDevice.objects.filter(name__in=names)
            patient_devices = models.PatientDevice.objects.bulk_create(
                [models.PatientDevice(patient=self.patient, device_id=d.id) for d in devices], 
                ignore_conflicts=True
            )
            result += patient_devices
        return result

    def refresh(self, *props: str) -> None:
        """Refreshes the properties passed as argument by reseting to None and
        fetching new data
        :param props:    List of properties to refresh from REFRESHABLE.
        :type props:     str
        """
        invalid = list(set(props) - self.REFRESHABLE)
        if invalid:
            raise ValueError(f"Invalid option: `{', '.join(invalid)}`")
        for prop in list(self.REFRESHABLE.intersection(props)):
            setattr(self, prop, None)
            getattr(self, prop)

    @transaction.atomic
    def update_in_orm(self, insight_ids=None, **kwargs) -> int:
        """Performs an update to each HealthInsight value in Patient's queryset
        :return:    Count of affected rows.
        :rtype:     int
        """
        if insight_ids is None:
            insight_ids = []
        count = 0
        insights_list = self.health_insights.filter(health_insight_id__in=insight_ids)
        if not insights_list.exists():
            insights_list = self.health_insights.all()  # All of Patient Insights
        try:
            for patient_insight in insights_list:
                patient_insight.collection_date = self.collection_date
                patient_insight.timezone = self.timezone
                health_insight = patient_insight.health_insight
                if health_insight.is_health_score():
                    patient_insight.set_health_score_value()
                else:  # Value is calculated from Redshift data
                    statistic = health_insight.statistic.name.lower()
                    # statistic needs to match Redshift calculated Agg Func (dict key)
                    apply_function = health_insight.get_devices_aggregation_function()
                    # apply_function defaults to max if db value of
                    # HealthInsight.in_devices_aggregation is inconsistent
                    result = self.get_in_values(
                        patient_insight, func=apply_function, agg=statistic, **kwargs
                    )
                    if result:
                        patient_insight.last_collection_time = result["latest_time"]
                        patient_insight.units = result["value_units"]
                        patient_insight.value = result[statistic]
                    else:
                        patient_insight.value = None
                patient_insight.save()
                count += 1
        except IntegrityError as e:
            raise RuntimeError("Values do not comply with ORM constraints") from e
        return count

    def process_with_sql(self, insight: models.PatientHealthInsight) -> dict:
        """Runs a SELECT aggregate for the function related to the Distributor
            Marker of the Health Insight passed as argument grouping by device.
        :return:    Calculated value of distributor markers grouped by device
        :rtype:     dict
        """
        health_insight = insight.health_insight
        agg_func = DistribMarkerStatistics.AggregateFunctions.get_agg_func(
            health_insight.statistic.name
        )
        distrib_marker = insight.get_distrib_marker()
        distrib_marker = distrib_marker.name if distrib_marker else None
        # Clear duplicates rows with distinct on value_timestamp_start
        # Most recent value is the one we want for each distribution marker and start dt
        no_dup = (
            "WITH no_dup as ("
            "   SELECT h.*, ROW_NUMBER() OVER (PARTITION BY distributor_marker, "
            "   value_timestamp_start ORDER BY created_at DESC) as rn"
            "   FROM health_standardized h "
            "   WHERE h.external_patient_id = %s AND h.distributor_marker = %s AND "
            "       h.value_timestamp_start >= %s AND h.value_timestamp_end <= %s) "
        )  # Do NOT quote %s -> SQL Injection risk
        no_overlap = (
            "WHERE t1.rn=1 AND NOT EXISTS("
            "  SELECT * FROM no_dup t2 "
            "  WHERE t1.value_timestamp_start > t2.value_timestamp_start "
            "    AND t1.value_timestamp_start < t2.value_timestamp_end "
            "    AND t1.device_name = t2.device_name "  # Allow different devices to overlap
            "    AND t1.value_timestamp_start != t2.value_timestamp_start "
            "    AND t1.value_timestamp_end != t2.value_timestamp_end ) "
        )
        query = (
            f"{no_dup}"
            f"SELECT device_name, {agg_func}, value_units, max(value_timestamp_end) as latest_time FROM no_dup t1 "
            f"{no_overlap}"
            " GROUP BY t1.device_name, t1.value_units;"
        )
        if health_insight.is_sleep_duration():
            query = (
                f"{no_dup} "
                "select device_name, sum(period_value), value_units, max(last_collection_time) as latest_time "
                "from ("
                f"  SELECT device_name, period, {agg_func} as period_value, "
                "    value_units, max(value_timestamp_end) as last_collection_time "
                "  FROM ("
                "    select device_name, value, value_units, value_timestamp_end, "
                "     diff, sum(CASE WHEN diff IS NULL OR diff > interval '1 HOUR' THEN 1 ELSE NULL END) "
                "      OVER (order by value_timestamp_end rows between unbounded preceding and current row) AS period"
                "    from ("
                "      SELECT t1.device_name, t1.value, t1.value_units, "
                "         t1.value_timestamp_start, t1.value_timestamp_end, "
                "         t1.value_timestamp_end - lag(t1.value_timestamp_end, 1) "
                "             OVER (ORDER by t1.value_timestamp_end) as diff"
                "      FROM no_dup t1 "
                f"{no_overlap}"
                "      ORDER BY t1.device_name, t1.value_timestamp_end)"
                "    order by device_name, value_timestamp_end "
                "  )"
                "  GROUP BY device_name, value_units, period "
                "  ORDER BY period) "
                f"where last_collection_time::date = '{insight.collection_date}' "
                "group by device_name, value_units;"
            )
         # list of dicts (one per device) with keys: device_name, value, value_units
        return self.redshift_execute_query(health_insight, distrib_marker, query)

    def process_with_python(self, insight: models.PatientHealthInsight) -> dict:
        health_insight = insight.health_insight
        distrib_marker = insight.get_distrib_marker()
        distrib_marker = distrib_marker.name if distrib_marker else None
        columns = [
            "device_name",
            "value",
            "value_units",
            "value_timestamp_start",
            "value_timestamp_end",
        ]
        cols = ", ".join(columns)
        query = (
            f"SELECT {cols} FROM health_standardized "
            "WHERE"
            "    external_patient_id = %s"
            "    AND device_name IS NOT NULL"
            "    AND distributor_marker = %s"
            "    AND value_timestamp_start >= %s"
            "    AND value_timestamp_end <= %s;"
        )
        rows = self.redshift_execute_query(
            health_insight, distrib_marker, query
        )
        df = pd.DataFrame.from_dict(rows).drop_duplicates()
        # TODO: remove overlaps? Sleep and Resting HR are the only ones that might need
        if df.empty:
            return []
        df = df.rename(columns={"value_timestamp_end": "latest_time"})
        grouped = df.groupby(by="device_name")["value"].agg(['sum','mean','max','min'])
        grouped = grouped.rename(columns={"mean": "avg"})
        grouped = grouped.reset_index()
        grouped["latest_time"] = df["latest_time"].max()
        grouped["value_units"] = df["value_units"].iloc[0]
        return grouped.to_dict("records") # list of dicts (one per device)

    def redshift_execute_query(self, health_insight, distrib_marker, query):
        """Executes a query on Redshift and returns the result as a list of dicts."""
        dt_from, dt_to = self.get_period_for_insight(health_insight)
        params = [
            self.ext_patient_id,
            distrib_marker,
            self.redshift.parse_dt(dt_from),
            self.redshift.parse_dt(dt_to),
        ]
        return self.redshift.execute(query, params, many=True, as_dict=True)
