import contextlib
import operator
import uuid
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import reduce
from statistics import mean

from celery import shared_task
from core.models import (
    BaseActivatorCategoryModel,
    BaseActivatorModel,
    BaseCategoryModel,
    BaseModel,
)
from dateutil import rrule
from dateutil.relativedelta import relativedelta
from django.conf import settings
from django.contrib.postgres.aggregates import ArrayAgg
from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator as MaxVal
from django.db import models
from django.utils import dateparse, timezone
from django_extensions.db.fields import AutoSlugField
from patients.models import Patient
from patients.services import get_gender_for_patient

from .redshift import DistribMarkerStatistics


class DataDevice(BaseModel):

    name = models.CharField(max_length=50, blank=False, null=False)
    version = models.CharField(max_length=16, blank=True, null=False)

    class Meta:
        unique_together = ["name", "version"]

    def __str__(self):
        return f"{self.name}"


class Statistic(BaseModel):
    AVG = DistribMarkerStatistics.AggregateFunctions.AVG.name
    MIN = DistribMarkerStatistics.AggregateFunctions.MIN.name
    MAX = DistribMarkerStatistics.AggregateFunctions.MAX.name
    SUM = DistribMarkerStatistics.AggregateFunctions.SUM.name

    OPTIONS = [(a.name, a.name) for a in DistribMarkerStatistics.AggregateFunctions]

    name = models.CharField(
        max_length=50, blank=False, null=False, choices=OPTIONS, unique=True
    )
    slug_name = AutoSlugField(
        populate_from="name", max_length=50, blank=False, null=False
    )

    def __str__(self):
        return f"{self.name}"

    @classmethod
    def get_or_create(cls, statistic):
        options = dict(cls.OPTIONS)
        try:  # validate name
            name = options[statistic]
        except KeyError as e:
            raise ValueError(
                f"`{statistic}` is not a valid statistic name. "
                f"Available options are `{', '.join(str(i) for i in options.values())}`"
            ) from e

        return cls.objects.get_or_create(name=name)

    @classmethod
    def from_redshift(
        cls,
        patient: int,
        d_marker: str,
        dt_from: datetime,
        dt_to: datetime,
    ) -> DistribMarkerStatistics:
        """Returns Patient DistMarkerStatistics pulled from Redshift."""
        return DistribMarkerStatistics(patient, d_marker, dt_from, dt_to)


class PatientDevice(BaseModel):

    patient = models.ForeignKey(
        Patient,
        on_delete=models.CASCADE,
        related_name="devices",
        blank=False,
        null=False,
    )
    device = models.ForeignKey(
        DataDevice,
        on_delete=models.PROTECT,
        related_name="patients",
        blank=False,
        null=False,
    )

    class Meta:
        unique_together = ["patient", "device"]

    def __str__(self):
        return f"{self.id}"


class DataDistributor(BaseModel):

    CLARITY_CORE = "CLARITY_CORE"

    name = models.CharField(max_length=50, blank=False, null=False, unique=True)
    base_date = models.DateField(null=False, blank=False, default=date.today)

    def __str__(self):
        return f"{self.name}"

    @classmethod
    def get_or_create_clarity_core(cls):
        return cls.objects.get_or_create(name=cls.CLARITY_CORE)


class DataDistributorLedger(BaseModel):
    DEF_BATCH_SIZE = int(settings.LEDGERS_BATCH_SIZE)  # Default batch size for ledgers
    DEF_TIME_FRAME = 24  # 24 hours for each ledger

    uuid = models.UUIDField(
        primary_key=False, unique=True, default=uuid.uuid4, editable=False
    )
    patient = models.ForeignKey(
        Patient,
        on_delete=models.CASCADE,
        related_name="data_distributor_ledgers",
        blank=False,
        null=False,
    )
    distributor = models.ForeignKey(
        DataDistributor,
        on_delete=models.PROTECT,
        related_name="patient_ledgers",
        blank=False,
        null=False,
    )
    started_at = models.DateTimeField(
        blank=False, null=False, validators=[MaxVal(timezone.now)]
    )
    finished_at = models.DateTimeField(
        blank=False, null=False, validators=[MaxVal(timezone.now)]
    )
    is_processed = models.BooleanField(
        default=False,
        blank=False,
        null=False,
    )
    processed_at = models.DateTimeField(
        blank=True, null=True, validators=[MaxVal(timezone.now)]
    )
    file_location = models.FilePathField(max_length=128, blank=True, null=True)

    def __str__(self):
        return f"{self.id}"

    def clean(self):
        super().clean()
        if self.started_at > self.finished_at:
            raise ValidationError({"finished_at": "finish must be after start"})
        if self.started_at.date() < self.distributor.base_date:
            d = self.distributor
            msg = f"Minimum start date for Distributor `{d.name}` is `{d.base_date}`"
            raise ValidationError({"started_at": msg})

    def save(self, *args, **kwargs) -> None:
        self.clean()  # Validates instance data prior to saving
        return super().save(*args, **kwargs)

    @staticmethod
    def _validate_ledgers_kwargs(start_dt, **kwargs):
        # Patient and DataDistributor are mandatory
        if len({"patient", "distributor"}.difference(kwargs)):
            raise ValueError("Expected kwargs Patient and DataDistributor")

        start_dt = start_dt.date() if isinstance(start_dt, datetime) else start_dt
        # Prevent Ledgers Older than Distributor base date
        if start_dt < kwargs["distributor"].base_date:
            d = kwargs["distributor"]
            raise ValueError(
                f"Minimum start date for Distributor `{d.name}` is `{d.base_date}`"
            )

    @classmethod
    def _batch_instances(
        cls,
        count: int,
        dt_start: datetime,
        t_frame=DEF_TIME_FRAME,
        freq=rrule.DAILY,
        interval=1,
        **kwargs,
    ):
        if count <= 0:
            return []

        kwargs.pop("started_at", None)
        kwargs.pop("finished_at", None)

        # Validate Ledger kwargs
        cls._validate_ledgers_kwargs(dt_start, **kwargs)
        dt_start = list(
            rrule.rrule(freq=freq, count=count, dtstart=dt_start, interval=interval)
        )

        finish_at = dt_start[0] + timedelta(hours=t_frame) - timedelta(seconds=1)
        dt_finish = list(
            rrule.rrule(freq=freq, count=count, dtstart=finish_at, interval=interval)
        )

        return [
            cls(started_at=dt_start[d], finished_at=dt_finish[d], **kwargs)
            for d in range(count)
        ]

    @classmethod
    def create_latest_ledger(cls, last_ledger=None, **kwargs):
        """Create and return latest entry after last ledger.
        Minimum time frame is 30 minutes past now.
        """
        if not last_ledger:
            last_ledger = (
                cls.objects.filter(
                    patient=kwargs["patient"], distributor=kwargs["distributor"]
                )
                .order_by("-started_at")
                .first()
            )
        dt_finish = timezone.now()  # timezone aware
        dt_start = dt_finish - timedelta(hours=cls.DEF_TIME_FRAME)
        empty_period = dt_finish - last_ledger.finished_at
        if (empty_period / timedelta(minutes=30)) < 1:
            return None  # Create no Ledger if less than 30 min from last ledger
        if dt_start <= last_ledger.finished_at:  # start 1 sec after last ledger
            dt_start = last_ledger.finished_at + timedelta(seconds=1)
        # Validate Ledger kwargs
        cls._validate_ledgers_kwargs(dt_start, **kwargs)
        return cls.objects.create(started_at=dt_start, finished_at=dt_finish, **kwargs)

    @classmethod
    def create_initial_ledgers(cls, count=30, start_at=None, **kwargs):
        """Creates past entries of 24h period using **kwargs.
        Created entries are returned.
        """
        # Prevent overflowing
        MAX_ENTRIES = 10 * int(cls.DEF_BATCH_SIZE)
        if not 0 < count < MAX_ENTRIES:
            raise ValueError(f"Maximum of {MAX_ENTRIES} entries")
        if not start_at:
            start_at = timezone.now()
        start_at -= timedelta(days=count)
        min_date = kwargs["distributor"].base_date
        if start_at.date() < min_date:
            start_at = min_date
        # Validate Ledger kwargs
        cls._validate_ledgers_kwargs(start_at, **kwargs)
        batch = cls._batch_instances(count, start_at, **kwargs)
        return cls.objects.bulk_create(batch)

    @classmethod
    def fill_in_ledgers(cls, dt_from: datetime, dt_to: datetime, hours=8, **kwargs):
        """Create unprocessed entries within time period and duration.
        Created entries are returned.
        """
        # Validate Ledger kwargs
        cls._validate_ledgers_kwargs(dt_from, **kwargs)
        # Prevent overflowing
        MAX_DATE = dt_from + timedelta(days=31)
        if dt_to > MAX_DATE:
            raise ValueError("Max period (dt_from -> dt_to) is 31 days")
        # Same Time Frame Ledgers
        SECONDS_IN_HOURS = 3600 * hours
        count_ledgers = int((dt_to - dt_from).total_seconds() // SECONDS_IN_HOURS)
        batch = cls._batch_instances(
            count_ledgers,
            dt_from,
            t_frame=hours,
            freq=rrule.HOURLY,
            interval=hours,
            **kwargs,
        )
        # Remaining period (Rest of time division)
        dt_start = dt_from
        if batch:
            dt_start = batch[-1].finished_at + timedelta(seconds=1)
        if dt_start < dt_to:
            batch.append(cls(started_at=dt_start, finished_at=dt_to, **kwargs))
        return cls.objects.bulk_create(batch)

    @classmethod
    def find_uuid(cls, **kwargs):
        """Returns a list of UUIDs from kwargs"""
        return cls.objects.filter(**kwargs).values_list("uuid", flat=True)

    @classmethod
    def get_ledger_batch(cls, batch_size=DEF_BATCH_SIZE, **kwargs):
        """
        Returns latest ledger entries for a given Patient and DataDistributor.
        Creates objects if necessary.
        """
        ledgers = cls.objects.filter(
            patient=kwargs["patient"], distributor=kwargs["distributor"]
        ).order_by("-started_at")
        if not ledgers.exists():
            cls.create_initial_ledgers(**kwargs)  # 30 initial ledgers of 24h
        last_ledger = ledgers.first()  # order_by -started_at
        latest_ledger = cls.create_latest_ledger(
            last_ledger=last_ledger, **kwargs
        )  # up to 24h past now (DEF_TIME_FRAME)
        min_date = kwargs["distributor"].base_date
        if latest_ledger:
            dt_from = last_ledger.finished_at + timedelta(seconds=1)
            dt_to = latest_ledger.started_at - timedelta(seconds=1)
            empty_period = dt_to - dt_from
            if empty_period.days > 30:  # Max period is 30 days
                dt_from = dt_to - timedelta(days=30)
                empty_period = dt_to - dt_from
            if empty_period / timedelta(seconds=1) >= 1 and dt_from.date() > min_date:
                cls.fill_in_ledgers(dt_from, dt_to, **kwargs)
        count_ledgers = ledgers.exclude(is_processed=True).count()
        if count_ledgers < batch_size:  # Create prior instances to complete Batch size
            dt_start = ledgers.last().started_at - timedelta(seconds=1)
            if dt_start.date() > min_date:
                prior_days = batch_size - count_ledgers
                cls.create_initial_ledgers(
                    count=prior_days, start_at=dt_start, **kwargs
                )
        return ledgers.exclude(is_processed=True)[:batch_size]

    @classmethod
    def mark_processed(cls, uuids: list, **kwargs):
        """Update rows with kwargs and set is_processed to True.
        Returns queryset of affected rows
        """
        kwargs["is_processed"] = True
        qs = cls.objects.filter(uuid__in=uuids)
        qs.update(**kwargs)
        return qs


class Marker(BaseModel):

    HEALTH_SCORE = "Health Score"
    SLEEP_DURATION = "Sleep Duration"

    name = models.CharField(max_length=50, blank=False, null=False, unique=True)

    def __str__(self):
        return f"{self.name}"


class DistributorMarker(BaseModel):

    marker = models.ForeignKey(
        Marker,
        on_delete=models.CASCADE,
        related_name="distributors",
        blank=False,
        null=False,
    )
    distributor = models.ForeignKey(
        DataDistributor,
        on_delete=models.CASCADE,
        related_name="markers",
        blank=False,
        null=False,
    )
    name = models.CharField(max_length=50, blank=False, null=False)

    class Meta:
        unique_together = ["distributor", "marker"]

    def __str__(self):
        return f"{self.name}"

    @classmethod
    def get_or_create_health_score(cls, name="Clarity Core Health Score"):
        return cls.objects.get_or_create(
            marker__name=Marker.HEALTH_SCORE,
            distributor__name=DataDistributor.CLARITY_CORE,
            name=name,
        )

    @classmethod
    def exists(cls, **kwargs) -> bool:
        return cls.objects.filter(**kwargs).exists()


class DistributorMarkerPair(DistributorMarker):
    """DistributorMarkerPair is a combination of DistributorMarker that is
    a DistributorMarker itself whose value will be defined by a function
    applied to the pair of DistributorMarker values.
    """

    OPERATIONS = (
        ("add", "add"),
        ("subtract", "subtract"),
        ("multiply", "multiply"),
        ("divide", "divide"),
    )
    first = models.ForeignKey(
        DistributorMarker, related_name="pair_1", null=False, on_delete=models.PROTECT
    )
    value_function = models.CharField(
        null=False, blank=False, choices=OPERATIONS, max_length=12
    )
    second = models.ForeignKey(
        DistributorMarker, related_name="pair_2", null=False, on_delete=models.PROTECT
    )
    units = models.CharField(max_length=25, blank=True)

    class Meta:
        unique_together = ["first", "second"]

    def get_value_function(self):
        operators = {
            "add": operator.add,
            "subtract": operator.sub,
            "multiply": operator.mul,
            "divide": operator.truediv,
        }
        if self.value_function not in operators:
            raise ValueError(f"Invalid value function {self.value_function}")
        return operators.get(self.value_function)

    def get_health_insight(self):
        if self.marker.health_insights.count() == 1:
            return self.marker.health_insights.first()
        raise ValueError(f"Invalid number of health insights for {self.marker.name}")

    @classmethod
    def create_patient_insights(cls, patient: int, collection_date: date, **kwargs):
        """Create or update derived patient insights from distributor markers
        combinations for a given patient and collection date.
        Return list of created or updated insights.
        """
        objs = []  # List of created or updated insights
        for pair in cls.objects.all():  # Check all combinations of markers
            derivable_insight = PatientHealthInsight.objects.filter(
                patient_id=patient,
                collection_date=collection_date,
                health_insight__marker__in=[pair.first.marker, pair.second.marker],
            ).exclude(value__isnull=True)
            # If both markers are present in collection date, create derived insight
            if derivable_insight.count() == 2:
                first = derivable_insight.get(health_insight__marker=pair.first.marker)
                second = derivable_insight.get(
                    health_insight__marker=pair.second.marker
                )
                value_function = pair.get_value_function()
                value = value_function(first.value, second.value)
                kwargs["value"] = value
                if pair.units:
                    kwargs["units"] = pair.units
                obj, created = PatientHealthInsight.objects.update_or_create(
                    patient_id=patient,
                    collection_date=collection_date,
                    health_insight=pair.get_health_insight(),
                    defaults=kwargs,
                )
                objs.append((obj, created))
        return objs


class HealthInsight(BaseModel):
    class ChartTypes(models.TextChoices):
        LINE = "line"
        BAR = "bar"

    # Python function to apply for multiple devices having data for the HealthInsight
    IN_DEVICES_AGGREGATION = (
        ("avg", "avg"),
        ("max", "max"),
        ("min", "min"),
        ("sum", "sum"),
    )

    marker = models.ForeignKey(
        Marker,
        on_delete=models.CASCADE,
        related_name="health_insights",
        blank=False,
        null=False,
    )
    statistic = models.ForeignKey(
        Statistic,
        on_delete=models.PROTECT,
        related_name="health_insights",
        blank=False,
        null=False,
    )
    default_duration = models.DurationField(
        null=False, blank=False, default=timedelta(hours=23, minutes=59, seconds=59)
    )
    start_from = models.TimeField(null=False, blank=False, default=time(00, 00))
    backwards = models.BooleanField(null=False, blank=False, default=False)
    in_devices_aggregation = models.CharField(
        null=False, blank=False, choices=IN_DEVICES_AGGREGATION, max_length=3
    )
    chart_type = models.CharField(
        max_length=50, choices=ChartTypes.choices, default=ChartTypes.LINE
    )

    def __str__(self):
        return f"{self.id} - {self.marker.name} - {self.statistic.name}"

    def get_devices_aggregation_function(self):
        # function needs to be applicable to list of decimals PatientHealthInsight.value
        return {"avg": mean, "max": max, "min": min, "sum": sum}.get(
            self.in_devices_aggregation, max
        )  # defaults to `max` if inconsistent in DB

    @classmethod
    def get_or_create_health_score(cls):
        return cls.objects.get_or_create(
            marker__name=Marker.HEALTH_SCORE,
            statistic__name=Statistic.MAX,
        )

    def get_period(self, collection_date: date, tz=timezone.utc) -> tuple:
        """Returns tuple of start and finish datetimes from collection date"""
        if not isinstance(collection_date, date):
            collection_date = dateparse.parse_date(collection_date)
        start_dt = end_dt = datetime.combine(collection_date, self.start_from)
        start_dt = timezone.make_aware(start_dt, timezone=tz).astimezone(
            tz=timezone.utc
        )  # From UTC to Timezone argument
        end_dt = timezone.make_aware(end_dt, timezone=tz).astimezone(tz=timezone.utc)
        if self.backwards:
            start_dt -= self.default_duration
        else:
            end_dt += self.default_duration
        return start_dt, end_dt

    def is_health_score(self):
        return self.marker.name == Marker.HEALTH_SCORE

    def is_sleep_duration(self):
        return self.marker.name == Marker.SLEEP_DURATION


class HealthInsightCategory(BaseCategoryModel):

    health_insights = models.ManyToManyField(HealthInsight, related_name="categories")

    def __str__(self):
        return f"{self.title}"


class Goal(BaseActivatorModel):
    """Goal is a target value for a set of PatientHealthInsight objects"""

    class Aggregations(models.TextChoices):
        """Aggregation to apply to a set of PatientHealthInsight values"""

        AVG = "avg"
        MAX = "max"
        MIN = "min"
        SUM = "sum"

    name = models.CharField(max_length=50, blank=False, null=False, unique=True)
    aggregation_type = models.CharField(
        max_length=16, blank=True, null=True, choices=Aggregations.choices
    )  # null = True do not perform aggregation, use the latest value
    # aggregation period is a timedelta from today to apply aggregation to
    aggregation_period = models.DurationField(
        blank=True, null=True, default=timedelta(days=30)
    )  # null = True do not perform aggregation, use the latest value
    # value should match PatientHealthInsight.value type (DecimalField) for aggregation
    value = models.DecimalField(decimal_places=4, max_digits=12, blank=True, null=True)
    value_unit = models.CharField(max_length=25, blank=True)

    def __str__(self):
        return f"{self.name}"

    def get_aggregate_function(self):
        """Returns function to apply to a queryset of PatientHealthInsight"""
        return {
            self.Aggregations.AVG: models.Avg,
            self.Aggregations.MAX: models.Max,
            self.Aggregations.MIN: models.Min,
            self.Aggregations.SUM: models.Sum,
        }.get(self.aggregation_type, None)

    def get_period(self, base_date: datetime = None):
        """Returns tuple of start and finish datetimes from base date
        defaulting to today.
        """
        if not base_date:
            base_date = timezone.now()
        return base_date - self.aggregation_period, base_date

    def get_value_for_patient(self, patient: int, base_date: datetime = None) -> float:
        """Returns calculated value for a patient based on the
        aggregation type of this Goal and base date.
        """
        patient_insights = PatientHealthInsight.objects.filter(
            patient=patient,
            health_insight__in_recommendations=self.recommendations.all(),
            collection_date__range=self.get_period(base_date),
        )
        if self.aggregation_type:
            return patient_insights.aggregate(
                result=self.get_aggregate_function()("value")
            )["result"]
        return patient_insights.latest("collection_date").value

    def eval_for_patient(self, patient_id: int, base_date: datetime = None) -> bool:
        """Evaluate this goal for a patient and collection date.
        if Goal.value is null or zero returns True if goal is achieved.
        else returns achieved percent of this goal.
        """
        patient_value = self.get_value_for_patient(patient_id, base_date)
        return patient_value / self.value if self.value else self.value == patient_value


class HealthInsightRecommendation(BaseActivatorCategoryModel):
    """Recommendations from a Goal for a set of HealthInsights for Patients"""

    goal = models.ForeignKey(
        Goal,
        on_delete=models.PROTECT,
        related_name="recommendations",
        null=False,
        blank=False,
    )
    health_insights = models.ManyToManyField(
        HealthInsight,
        related_name="in_recommendations",
        help_text=(
            "Set of HealthInsights that are used to generate this recommendation. "
            "Don't mix value types and units"
        ),
    )
    priority = models.PositiveSmallIntegerField(blank=False, null=False, default=1)
    delivery_time = models.TimeField(null=True)
    notification_text = models.TextField(blank=True, null=True)
    alt_notification_text = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ["goal", "priority"]
        unique_together = ["goal", "priority"]

    def __str__(self):
        return f"{self.id}"

    def get_alt_notification(self, *args) -> str:
        """Returns the Alt Notification text for this Recommendation"""
        return self.alt_notification_text.format(*args)

    def get_markers_names(self):
        markers = self.health_insights.values_list("marker__name", flat=True).distinct()
        return ", ".join(markers)

    def get_notification(self, *args) -> str:
        """Returns the Notification text for this Recommendation"""
        return self.notification_text.format(*args)

    def get_patient_insights(self, patient_id: int) -> models.QuerySet:
        """Returns all Patient Health Insights for this recommendation within
        the goal aggregation period, oldest collection first.
        """
        return PatientHealthInsight.objects.filter(
            patient=patient_id,
            health_insight__in=self.health_insights.all(),
            collection_date__range=self.goal.get_period(),
        ).order_by("collection_date")


class DeviceSupportedHealthInsight(BaseModel):

    device = models.ForeignKey(
        DataDevice,
        on_delete=models.PROTECT,
        related_name="health_insights",
        blank=False,
        null=False,
    )
    health_insight = models.ForeignKey(
        HealthInsight,
        on_delete=models.CASCADE,
        related_name="devices",
        blank=False,
        null=False,
    )
    priority = models.PositiveSmallIntegerField(blank=True, null=True)

    def __str__(self):
        return f"{self.id}"


class PatientHealthInsight(BaseModel):

    TIMEZONES = tuple(zip(timezone.pytz.all_timezones, timezone.pytz.all_timezones))
    PERIOD_AGG = {
        "quarter": models.functions.ExtractQuarter,
        "month": models.functions.ExtractMonth,
        "week": models.functions.ExtractWeek,
        "day": models.functions.ExtractDay,
        "week_day": models.functions.ExtractWeekDay,
    }

    health_insight = models.ForeignKey(
        HealthInsight,
        on_delete=models.PROTECT,
        related_name="patients",
        blank=False,
        null=False,
    )
    patient = models.ForeignKey(
        Patient,
        on_delete=models.CASCADE,
        related_name="health_insights",
        blank=False,
        null=False,
    )
    data_distributor = models.ForeignKey(
        DataDistributor,
        on_delete=models.PROTECT,
        related_name="patients_health_insights",
        blank=True,
        null=True,
    )
    value = models.DecimalField(decimal_places=4, max_digits=12, blank=True, null=True)
    collection_date = models.DateField(
        blank=False, null=False, validators=[MaxVal(timezone.localdate)]
    )
    last_collection_time = models.TimeField(blank=False, null=True)
    units = models.CharField(max_length=25, blank=True)
    timezone = models.CharField(
        max_length=50, blank=False, default="UTC", choices=TIMEZONES
    )

    class Meta:
        unique_together = ["health_insight", "patient", "collection_date"]

    def __str__(self):
        return f"{self.id}"

    @property
    def display(self):
        """True if this insight has and ID and a value"""
        return (self.id and self.value) is not None

    @property
    def state(self):
        return HealthInsightStateValue.get_state(self)

    def get_distrib_marker(self) -> DistributorMarker:
        return self.health_insight.marker.distributors.filter(
            distributor=self.data_distributor
        ).first()

    def get_devices(self, name_only=False) -> list:
        patient_devices = self.health_insight.devices.filter(
            device__patients__patient=self.patient
        ).order_by("priority")
        if name_only:
            return patient_devices.values_list("device__name", flat=True).distinct()
        return patient_devices

    @classmethod
    def update_or_create_health_score(cls, patient_id, **kwargs):
        try:
            kwargs["patient"] = Patient.objects.get(id=patient_id)
        except Patient.DoesNotExist as e:
            raise ValueError("Invalid Patient ID") from e
        kwargs["health_insight"], _ = HealthInsight.get_or_create_health_score()
        kwargs["data_distributor"], _ = DataDistributor.get_or_create_clarity_core()
        if "collection_date" not in kwargs:
            kwargs["collection_date"] = timezone.localdate()
        params = {  # Filter params based on unique_together constraints
            "patient": kwargs["patient"],
            "health_insight": kwargs["health_insight"],
            "collection_date": kwargs["collection_date"],
            "defaults": kwargs,  # update if matching record exists for this params
        }
        obj, _ = cls.objects.update_or_create(**params)
        return obj, _

    def set_health_score_value(self) -> None:
        from scoring.calculators import wh_score_details_for_patient

        if not self.health_insight.is_health_score():
            return
        try:
            score, _ = wh_score_details_for_patient(self.patient.id, run_score=False)
            if score is None:
                score, _ = wh_score_details_for_patient(self.patient.id, run_score=True)
            value = score.score
        except Exception as e:
            print(e)
            value = None
        self.value = value
        self.save()

    @staticmethod
    def clean_period_avg_interval(period, start, end) -> None:
        """Validates period and start/end dates for period average"""
        period_diff = relativedelta(end, start)
        remain_year_or_month = period_diff.years > 0 or period_diff.months > 0
        max_interval = [
            (period != "year" and period_diff.years > 0),
            (period == "day" and (period_diff.days > 30 or remain_year_or_month)),
        ]  # Validate diff period according to agg options (avoid repeating values error)
        if any(max_interval):
            raise ValueError(
                f"Exceeded max interval for period option `{period}` diff `{period_diff}`"
            )

    @classmethod
    def period_avg(
        cls, patient: int, health_insight: int, from_date: date, to_date: date, **params
    ) -> models.QuerySet:
        """Returns Queryset of avg value and period within the given date range
        :param from_date: Start date of the period.
        :param to_date: End date of the period.
        :param params: Additional params to handle the queryset.
        :param qs: Initial Queryset to filter period range. Default = `filter(patient)`
        :param period: Option for the Queryset aggregation. Default = `day`.
        """
        qs = params.get("qs", cls.objects.filter(patient=patient))
        period = params.get("period", "day")  # default is per day
        if period not in cls.PERIOD_AGG:
            raise ValueError(f"Invalid period. Choose from {cls.PERIOD_AGG.keys()}")
        if not isinstance(to_date, date):
            to_date = dateparse.parse_date(to_date)
        if not isinstance(from_date, date):
            from_date = dateparse.parse_date(from_date)
        cls.clean_period_avg_interval(period, from_date, to_date)
        tz = to_date.tzinfo
        date_extract = {
            "year": models.functions.ExtractYear("collection_date", tzinfo=tz),
            period: cls.PERIOD_AGG[period]("collection_date", tzinfo=tz),
        }  # year for reference to assign base_date
        filter_params = {
            "health_insight": health_insight,
            "collection_date__range": (from_date, to_date),
        }
        return (
            qs.filter(**filter_params)
            .values(**date_extract)
            .annotate(avg=models.Avg("value"))
        )


class HealthInsightStateValue(BaseModel):
    class States(models.TextChoices):
        UNKNOWN = "Unknown"
        RISK = "Risk"
        WARNING = "Warning"
        SUCCESS = "Success"

    class Genders(models.TextChoices):
        ALL = "All"
        FEMALE = "Female"
        MALE = "Male"

    PERCENT_RANGE = {  # Ranges must be exclusive
        States.RISK: (0, 49),
        States.WARNING: (51, 75),
        States.SUCCESS: (76, 100),
    }

    health_insight = models.ForeignKey(
        HealthInsight,
        on_delete=models.CASCADE,
        related_name="states",
        blank=False,
        null=False,
    )
    applies_to_gender = models.CharField(
        max_length=16,
        choices=Genders.choices,
        blank=False,
        null=False,
        default=Genders.ALL,
    )
    state = models.CharField(
        max_length=16, choices=States.choices, blank=False, null=False
    )
    # inclusive limits of value range -> PatientHealthInsight.value (same field type)
    upper_limit = models.DecimalField(
        decimal_places=4, max_digits=12, blank=True, null=True
    )
    lower_limit = models.DecimalField(
        decimal_places=4, max_digits=12, blank=True, null=True
    )
    reversed_cumulative = models.BooleanField(default=False)  # Resting HR

    class Meta:
        unique_together = [
            ("health_insight", "upper_limit", "lower_limit"),
            ("health_insight", "state", "applies_to_gender"),
        ]

    @classmethod
    def get_obj_from_patient_insight(cls, patient_insight: PatientHealthInsight):
        """Query HealthInsightStateValue based on value ranges for the HealthInsight
        :param patient_insight: PatientHealthInsight instance to get state for.
        :return: HealthInsightStateValue instance.
        """
        if not patient_insight or patient_insight.value is None:
            return None  # No value to compare
        filter_params = {
            "health_insight": patient_insight.health_insight,
            "applies_to_gender": cls.Genders.ALL,
        }

        filter_gender = patient_insight.health_insight.states.exclude(
            applies_to_gender=cls.Genders.ALL
        ).exists()
        if filter_gender:  # Resting HR - Male/Female Specific
            gender = get_gender_for_patient(patient_insight.patient.id)
            # Won't find value if portal doesn't send gender
            if gender in cls.Genders.values:  # if valid use to specify query
                filter_params["applies_to_gender"] = gender
        h_insight_state = (
            cls.objects.filter(**filter_params)
            .filter(
                models.Q(upper_limit__gte=patient_insight.value)
                | models.Q(lower_limit__lte=patient_insight.value)
            )
            .exclude(
                models.Q(lower_limit__gt=patient_insight.value)
                | models.Q(upper_limit__lt=patient_insight.value)
            )
        )
        return h_insight_state.first()

    @classmethod
    def get_state(cls, patient_insight: PatientHealthInsight) -> str:
        """Fetches HealthInsightStateValue for this PatientHealthInsight instance
        :param patient_insight: PatientHealthInsight instance to get state for.
        :return: Dict State and Percentage of the PatientHealthInsight.
        """
        obj = cls.get_obj_from_patient_insight(patient_insight)
        if not obj:
            return {"name": cls.States.UNKNOWN, "percent_value": None}
        try:
            lower_limit = obj.lower_limit or 0
            upper_limit = obj.upper_limit or 0
            state_range = abs(upper_limit - lower_limit)
            lowbound, highbound = cls.PERCENT_RANGE[obj.state]  # 0-49%, 50-74%, 75-100%
            perc_range = highbound - lowbound
            if obj.reversed_cumulative:
                if upper_limit:
                    proportion = abs(patient_insight.value - upper_limit) / state_range
                    percentage = round(lowbound + float(perc_range * proportion))
                else:
                    proportion = abs(patient_insight.value - lower_limit) / state_range
                    percentage = round(highbound - float(perc_range * proportion))
            elif upper_limit:
                proportion = abs(patient_insight.value - upper_limit) / state_range
                percentage = round(highbound - float(perc_range * proportion))
            else:
                proportion = abs(patient_insight.value - lower_limit) / state_range
                percentage = round(lowbound + float(perc_range * proportion))
            # keep perc within bounds
            if percentage > highbound:
                percentage = highbound
            elif percentage < lowbound:
                percentage = lowbound
            percentage /= 100
        except ZeroDivisionError:
            percentage = None
        return {"name": obj.state, "percent_value": percentage}

    def clean(self):
        if not self.upper_limit and not self.lower_limit:
            msg = "At least one of upper limit or lower limit must be specified"
            raise ValidationError({"upper_limit": msg})
        with contextlib.suppress(TypeError):  # if None
            if self.upper_limit < self.lower_limit:
                msg = "Upper limit must be greater than lower limit"
                raise ValidationError({"upper_limit": msg})
        wrong_limits = []
        # Check bad combination of ranges with registered entries
        if self.upper_limit:
            if self.lower_limit:
                wrong_limits = [
                    models.Q(
                        upper_limit__lte=self.upper_limit,
                        lower_limit__gte=self.upper_limit,
                    ),
                    models.Q(
                        upper_limit__lte=self.lower_limit,
                        lower_limit__gte=self.lower_limit,
                    ),
                ]
            wrong_limits.append(
                models.Q(upper_limit__lte=self.upper_limit, lower_limit__isnull=True)
            )
        if self.lower_limit:
            wrong_limits.append(
                models.Q(upper_limit__isnull=True, lower_limit__gte=self.lower_limit)
            )
        invalid_range = (
            HealthInsightStateValue.objects.filter(
                models.Q(health_insight=self.health_insight),
                models.Q(applies_to_gender=self.applies_to_gender),
            )
            .filter(reduce(operator.or_, wrong_limits))
            .exclude(id=self.id)
        )
        if invalid_range.exists():
            msg = (
                "Other State value is already registered within these limits "
                "of Health Insight value and gender option"
            )
            raise ValidationError({"health_insight": msg})

    def save(self, *args, **kwargs) -> None:
        self.clean()  # Validates instance data prior to saving
        return super().save(*args, **kwargs)


class PatientHealthInsightRecommendation(BaseActivatorModel):
    """Patient Health Insight(s) Recommendation"""

    patient = models.ForeignKey(
        Patient,
        on_delete=models.CASCADE,
        related_name="health_insights_recommendations",
        blank=False,
        null=False,
    )
    recommendation = models.ForeignKey(
        HealthInsightRecommendation,
        on_delete=models.CASCADE,
        related_name="patients_recommendations",
        blank=False,
        null=False,
    )

    def __str__(self):
        return f"{self.patient} - {self.recommendation}"

    @staticmethod
    def get_for_patient(patient_id: int, active_only: bool = True) -> models.QuerySet:
        """Returns Queryset of Recommendation for the given patient_id"""
        params = {"patients_recommendations__patient": patient_id}
        ACTIVE = PatientHealthInsightRecommendation.ACTIVE_STATUS
        INACTIVE = PatientHealthInsightRecommendation.INACTIVE_STATUS
        if active_only:
            params["patients_recommendations__status"] = ACTIVE
        qs = (
            HealthInsightRecommendation.objects.filter(**params)
            .values("goal", "title", "slug", "description")
            .annotate(
                id=models.F("patients_recommendations__id"),
                status=models.Case(
                    models.When(status=ACTIVE, then=models.Value("Active")),
                    models.When(status=INACTIVE, then=models.Value("Inactive")),
                    output_field=models.CharField(),
                ),
                from_insights=ArrayAgg("health_insights"),
            )
        )
        return qs.distinct().order_by("-modified")  # most recent updated first


class HealthInsightDisplayRuleSet(BaseModel):
    """Priority set of Comparison Rules for filtering PatientHealthInsight."""

    health_insight = models.ForeignKey(
        HealthInsight,
        on_delete=models.CASCADE,
        related_name="disp_rule_set",
        blank=False,
        null=False,
    )
    priority = models.PositiveSmallIntegerField(blank=False, null=False)
    name = models.CharField(max_length=25, blank=False, null=False)
    show_null = models.BooleanField(default=False)

    def eval_insight(self, insight: PatientHealthInsight, eval=True) -> bool:
        """Evaluate all Comparison Rules for the Health Insight"""
        return all(display.eval(insight) == eval for display in self.rules.all())

    def filter_eval(self, insight: PatientHealthInsight, eval=True) -> filter:
        """Filter related Rules by its evaluation provided as argument"""
        return filter(lambda display: display.eval(insight) == eval, self.rules.all())

    def get_for_patient_display(self, patient_id: int) -> PatientHealthInsight:
        """Returns the latest PatientHealthInsight that eval all Rules to true
        or an unsaved PatientHealthInsight obj if none found
        """
        Comparators = HealthInsightDisplayRule.Comparator
        expressions = {
            Comparators.EQUAL: "__exact",
            Comparators.LESS_THAN: "__lt",
            Comparators.LESS_THAN_EQUAL: "__lte",
            Comparators.GREATER_THAN: "__gt",
            Comparators.GREATER_THAN_EQUAL: "__gte",
        }
        rules = HealthInsightDisplayRule.objects.filter(
            rule_set__health_insight=self.health_insight
        ).order_by("rule_set__priority")
        # loop all rules of all HealthInsightDisplayRuleSet of this HealthInsight
        filters = {}  # compose filters for PatientHealthInsight queryset
        for rule in rules.values("field", "comparator", "comparison_value").distinct():
            if rule["field"] == "collection_date":
                days_ago = int(rule["comparison_value"])
                value = timezone.localdate() - timedelta(days=days_ago)
            else:
                value = rule["comparison_value"]
            comparator = expressions[rule["comparator"]]
            filters[f"{rule['field']}{comparator}"] = value
        patient_insights = PatientHealthInsight.objects.filter(
            patient_id=patient_id, health_insight=self.health_insight, **filters
        ).order_by("-collection_date")
        if not self.show_null:
            patient_insights.exclude(value__isnull=True).first()
        insight = patient_insights.first()
        if insight is not None:
            return insight
        distrib_clarity_core, _ = DataDistributor.get_or_create_clarity_core()
        return PatientHealthInsight(
            health_insight=self.health_insight,
            patient_id=patient_id,
            data_distributor=distrib_clarity_core,
            collection_date=None,
            last_collection_time=None,
        )

    class Meta:
        ordering = ["health_insight", "priority"]
        unique_together = ["health_insight", "priority"]

    def __str__(self):
        return f"{self.name}"


class HealthInsightDisplayRule(BaseModel):
    """Comparison Rules for HealthInsightDisplayRuleSet."""

    class Comparator(models.TextChoices):
        """Parsing symbols used in Comparisons."""

        EQUAL = "=="
        LESS_THAN = "<"
        LESS_THAN_EQUAL = "<="
        GREATER_THAN = ">"
        GREATER_THAN_EQUAL = ">="

    class PatientHealthInsightField(models.TextChoices):
        """Field in PatientHealthInsight model - used for comparison."""

        VALUE = "value"
        COLLECTION_DATE = "collection_date"

    class ComparisonValueType(models.TextChoices):
        """Type of comparison value compatible with the PatientHealthInsight.value"""

        DECIMAL = "decimal"
        DAYS_AGO = "days"

    rule_set = models.ForeignKey(
        HealthInsightDisplayRuleSet,
        on_delete=models.CASCADE,
        related_name="rules",
        blank=False,
        null=False,
    )
    field = models.CharField(
        choices=PatientHealthInsightField.choices,
        max_length=16,
        blank=False,
        null=False,
    )
    comparator = models.CharField(
        max_length=2, blank=False, null=False, choices=Comparator.choices
    )
    comparison_value = models.CharField(max_length=16, blank=False, null=False)
    comparison_value_type = models.CharField(
        max_length=8,
        blank=False,
        null=False,
        choices=ComparisonValueType.choices,
        default=ComparisonValueType.DECIMAL,
    )

    def __str__(self):
        return f"{self.id}"

    def eval(self, insight: PatientHealthInsight) -> bool:
        """Generate and evalute Python comparison expression"""
        return eval(self.get_expression(insight)) if insight.collection_date else False

    def get_expression(self, insight) -> str:
        """Returns the Python expression to be evaluated"""
        field_value = self.get_field_value(insight)
        return f"{field_value} {self.comparator} {self.comparison_value}"

    def get_field_value(self, insight: PatientHealthInsight) -> float:
        if self.comparison_value_type != self.ComparisonValueType.DAYS_AGO:
            return getattr(insight, self.field)
        from_date = timezone.localdate(
            timezone=timezone.pytz.timezone(insight.timezone)
        )
        return (from_date - insight.collection_date).days

    def clean(self):
        collec_date = self.PatientHealthInsightField.COLLECTION_DATE
        collec_rule = self.rule_set.rules.filter(field=collec_date).exclude(id=self.id)
        if self.field == collec_date and collec_rule.exists():
            raise ValidationError(
                {"field": ["Collection date Rule already exists for this RuleSet"]}
            )
        if self.comparison_value_type == self.ComparisonValueType.DAYS_AGO:
            try:
                int(self.comparison_value)
            except ValueError as e:
                raise ValidationError(
                    {"comparison_value": ["Must be an integer for days ago"]}
                ) from e
            if self.field != collec_date:
                msg = "Use 'Collection Date' for comparison value type 'Days Ago'"
                raise ValidationError({"field": [msg]})

    def save(self, *args, **kwargs) -> None:
        self.clean()  # Validates instance data prior to saving
        return super().save(*args, **kwargs)


class RecommendationTrigger(BaseActivatorCategoryModel):
    """Triggers for HealthInsightRecommendation."""

    DAYS_SINCE = 2  # Days since Insight value < goal

    class Markers(models.TextChoices):
        """Markers to use in personalized proxy Recommendation Triggers"""

        CALORIES = "Calories Burned"
        HEALTH_SCORE = "Health Score"
        RESTING_HEART_RATE = "Resting Heart Rate"
        SLEEP = "Sleep Duration"
        STEPS = "Steps"
        WEIGHT = "Weight"

    class ProxyTriggers(models.TextChoices):
        """Proxy Triggers to set the processing behavior of a Recommendation
        these are the model proxies currently supported.
        """

        CALORIES = "TriggerCalories"
        RESTING_HEART_RATE = "TriggerRestingHR"
        SLEEP = "TriggerSleep"
        STEPS = "TriggerSteps"
        WEIGHT = "TriggerWeight"

    recommendation = models.ForeignKey(
        HealthInsightRecommendation,
        on_delete=models.CASCADE,
        related_name="triggers",
        blank=False,
        null=False,
    )
    proxy_trigger = models.CharField(
        max_length=32, blank=False, null=False, choices=ProxyTriggers.choices
    )

    def __init__(self, *args, **kwargs):
        """Sets the model class to the proxy_trigger if set."""
        super().__init__(*args, **kwargs)
        if self.proxy_trigger:
            self.__class__ = self.get_trigger()

    @property
    def goal(self):
        return self.recommendation.goal

    def get_patient_insights(self, patient_id: int, **kwargs) -> models.QuerySet:
        """Returns PatientHealthInsight Queryset for a Trigger
        :param patient_id: Patient ID
        :param kwargs: Other parameters to filter on
        """
        return PatientHealthInsight.objects.filter(patient_id=patient_id, **kwargs)

    def get_date_range(self) -> list:
        """Returns the date range to check when processing a Trigger.
        :return: list of [start_date, end_date]
        """
        today = timezone.localdate()
        return [
            today - timedelta(days=self.DAYS_SINCE + 1),
            today - timedelta(days=self.DAYS_SINCE - 1),  # Exclude current day
        ]

    def get_trigger(self):
        """Returns the model class Trigger for its proxy_trigger"""
        trigger_mapping = {
            RecommendationTrigger.ProxyTriggers.STEPS: TriggerSteps,
            RecommendationTrigger.ProxyTriggers.CALORIES: TriggerCalories,
            RecommendationTrigger.ProxyTriggers.SLEEP: TriggerSleep,
            RecommendationTrigger.ProxyTriggers.RESTING_HEART_RATE: TriggerRestingHR,
            RecommendationTrigger.ProxyTriggers.WEIGHT: TriggerWeight,
        }
        return trigger_mapping[self.proxy_trigger]

    def process(self, patient_id: int) -> tuple:
        """Process a Trigger for a Patient."""
        raise NotImplementedError("Must be implemented by child class")

    @shared_task
    def process_all(patient_id: int) -> list:
        """Process all available Triggers for a Patient.
        :param patient_id: Patient
        Return list of tuple of (object, created) for every Recommendation Trigger
        """
        # triggers are processed as child proxy classes
        all_triggers = RecommendationTrigger.objects.all()
        return [trigger.process(patient_id) for trigger in all_triggers]

    def update_or_create(self, patient_id: int) -> tuple:
        """Update or create a PatientHealthInsightRecommendation for a Patient.
        :param patient_id: Patient ID
        Return tuple of (object id, created bool)
        """
        active = PatientHealthInsightRecommendation.ACTIVE_STATUS
        try:
            patient = Patient.objects.get(id=patient_id)
        except Patient.DoesNotExist as e:
            raise ValueError("Invalid Patient ID") from e
        obj, created = PatientHealthInsightRecommendation.objects.update_or_create(
            patient=patient,
            recommendation=self.recommendation,
            defaults={"status": active, "activate_date": timezone.now()},
        )
        return (obj.id, created)


class TriggerSteps(RecommendationTrigger):
    """Trigger for HealthInsightRecommendation based on Steps Marker"""

    class Meta:
        proxy = True

    def process(self, patient_id: int) -> tuple:
        """If >/= `n` days since steps < goal (does not include current day)
        then activate or create a PatientHealthInsightRecommendation for today.
        :param patient_id: Patient ID
        Return tuple of (object, created)
        """
        today = timezone.localdate()
        kwargs = {  # Kwargs to pass to get_patient_insights
            "health_insight__marker__name": self.Markers.STEPS,
            "collection_date__range": (today - timedelta(days=30), today),  # 30 days
        }
        qs = super().get_patient_insights(patient_id, **kwargs)
        avg = qs.aggregate(models.Avg("value"))["value__avg"]
        if not avg:
            return None, False  # No object, not created/updated
        date_range = self.get_date_range() # Date range to check for below average
        below_avg = qs.filter(collection_date__range=date_range, value__lt=avg)
        if avg <= self.goal.value or below_avg.count() == self.DAYS_SINCE:
            return super().update_or_create(patient_id)
        return None, False


class TriggerCalories(RecommendationTrigger):
    """Trigger for HealthInsightRecommendation based on Calories Marker"""

    class Meta:
        proxy = True

    def process(self, patient_id: int) -> tuple:
        """If >/= `n` days since Active Energy > goalcals (does not include current day)
        then activate or create a PatientHealthInsightRecommendation for today.
        :param patient_id: Patient ID
        Return tuple of (object, created)
        """
        today = timezone.localdate()
        kwargs = {  # Kwargs to pass to get_patient_insights
            "health_insight__marker__name": self.Markers.CALORIES,
            "collection_date__range": (today - timedelta(days=30), today),  # 30 days
        }
        qs = super().get_patient_insights(patient_id, **kwargs)
        avg = qs.aggregate(models.Avg("value"))["value__avg"]  # 30 days avg
        if not avg:
            avg = 0
        date_range = self.get_date_range()  # Date range to check for above average
        above_avg = qs.filter(collection_date__range=date_range, value__gt=avg)
        if avg <= self.goal.value or above_avg.count() == self.DAYS_SINCE:
            return super().update_or_create(patient_id)
        return None, False  # No object, not created/updated


class TriggerSleep(RecommendationTrigger):
    """Trigger for HealthInsightRecommendation based on Sleep Marker"""

    class Meta:
        proxy = True

    def process(self, patient_id: int) -> tuple:
        """If 10% or more < goal sleep hours
        then activate or create a PatientHealthInsightRecommendation for today.
        :param patient_id: Patient ID
        Return tuple of (object, created)
        """
        kwargs = {  # Kwargs to pass to get_patient_insights
            "health_insight__marker__name": self.Markers.SLEEP,
            "collection_date": timezone.localdate(),  # latest sleep = today
        }
        latest = super().get_patient_insights(patient_id, **kwargs).last()
        if not latest:
            return None, False  # No object, not created/updated
        value = latest.value or 0
        if value < self.goal.value * Decimal(0.9):  # < 90% goal
            return super().update_or_create(patient_id)
        return None, False  # No object, not created/updated


class TriggerRestingHR(RecommendationTrigger):
    """Trigger for HealthInsightRecommendation based on Resting HR Marker"""

    class Meta:
        proxy = True

    def process(self, patient_id: int) -> tuple:
        """>10% above training average
        then activate or create a PatientHealthInsightRecommendation for today.
        :param patient_id: Patient ID
        Return tuple of (object, created)
        """
        today = timezone.localdate()
        kwargs = {  # Kwargs to pass to get_patient_insights
            "health_insight__marker__name": self.Markers.RESTING_HEART_RATE,
            "collection_date__range": (today - timedelta(days=30), today),  # 30 days
        }
        qs = super().get_patient_insights(patient_id, **kwargs)
        avg = qs.aggregate(models.Avg("value"))["value__avg"]  # 30 days avg
        if not avg:
            return None, False  # No object, not created/updated
        # Checks if latest is > 10% avg
        above_avg = qs.filter(collection_date=today, value__gt=avg * Decimal(1.1))
        return super().update_or_create(patient_id) if above_avg else (None, False)


class TriggerWeight(RecommendationTrigger):
    """Trigger for HealthInsightRecommendation based on Weight Marker"""

    class Meta:
        proxy = True

    def process(self, patient_id: int) -> tuple:
        """If above or below goal weight
        then activate or create a PatientHealthInsightRecommendation for today.
        :param patient_id: Patient ID
        Return tuple of (object, created)
        """
        kwargs = {"health_insight__marker__name": self.Markers.WEIGHT}
        weight = super().get_patient_insights(patient_id, **kwargs).last()
        if weight and weight.value and int(weight.value) != int(self.goal.value):
            # Compare int value for weight
            return super().update_or_create(patient_id)
        return None, False  # No object, not created/updated
