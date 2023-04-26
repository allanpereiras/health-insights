"""
Serializers allow complex data such as querysets and model instances to be
converted to native Python datatypes to be rendered into JSON, XML or other.

Serializers also provide deserialization, allowing parsed data to be converted
back into complex types, after first validating the incoming data.

Serialization in REST framework is a two-phase process:
    1. Serializers marshal between complex types like model instances, and
    python primitives.
    2. The process of marshalling between python primitives and request and
    response content is handled by parsers and renderers.
"""

from datetime import date, datetime, time, timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta
from django.db.models import Avg, F, Max, Min, Sum
from django.urls import reverse
from django.utils import dateparse, timezone
from insights import models
from patients import utils
from patients.utils import get_or_create_patient_with_external_id as get_patient
from rest_framework import exceptions, serializers


class ExternalPatientSerializerField(serializers.Field):
    """Serializer Field to expose External Patient instead of Patient."""

    def to_representation(self, patient):
        external_patient = utils.get_clarity_portal_external_patient_for_patient(
            patient
        )
        if external_patient is not None:
            return external_patient.external_id
        return None

    def to_internal_value(self, external_id):
        return utils.get_or_create_patient_with_external_id(external_id)


class DataDistributorLedgerSerializer(serializers.ModelSerializer):
    """
    ModelSerializer for Data Distributor Ledgers.

    Patient is exposed as an External Patient using a Serializer Field
    attr:`~external_patient_id` see :class:`~ExternalPatientSerializerField`
    """

    external_patient_id = ExternalPatientSerializerField(
        source="patient", read_only=True
    )
    file_location = serializers.CharField()

    class Meta:
        model = models.DataDistributorLedger
        fields = [
            "uuid",
            "external_patient_id",
            "distributor",
            "started_at",
            "finished_at",
            "is_processed",
            "processed_at",
            "file_location",
        ]


class MarkerSerializer(serializers.ModelSerializer):
    """ModelSerializer for Marker"""

    class Meta:
        model = models.Marker
        fields = ["id", "name"]


class StatisticSerializer(serializers.ModelSerializer):
    """ModelSerializer for Statistic"""

    class Meta:
        model = models.Statistic
        fields = ["id", "name", "slug_name"]


class HealthInsightSerializer(serializers.ModelSerializer):
    """ModelSerializer for Health Insight"""

    marker = MarkerSerializer(many=False, read_only=True)
    statistic = StatisticSerializer(many=False, read_only=True)
    categories = serializers.SerializerMethodField()

    class Meta:
        model = models.HealthInsight
        fields = [
            "id",
            "marker",
            "statistic",
            "chart_type",
            "default_duration",
            "start_from",
            "backwards",
            "categories",
        ]

    def get_categories(self, obj):
        return obj.categories.values("id", "title")


class PatientHealthInsightSerializer(serializers.ModelSerializer):
    """ModelSerializer for Patient Health Insight"""

    external_patient_id = ExternalPatientSerializerField(
        source="patient", read_only=True
    )
    collection_date = serializers.SerializerMethodField()
    chart_type = serializers.CharField(
        source="health_insight.chart_type", read_only=True
    )
    health_insight_categories = serializers.SerializerMethodField()
    marker_id = serializers.IntegerField(
        source="health_insight.marker.id", read_only=True
    )
    marker_name = serializers.CharField(
        source="health_insight.marker.name", read_only=True
    )
    statistic_name = serializers.CharField(
        source="health_insight.statistic.name", read_only=True
    )
    data_distributor = serializers.CharField(
        source="data_distributor.name", read_only=True
    )

    class Meta:
        model = models.PatientHealthInsight
        fields = [
            "id",
            "health_insight_id",
            "collection_date",
            "chart_type",
            "health_insight_categories",
            "external_patient_id",
            "marker_id",
            "marker_name",
            "statistic_name",
            "value",
            "state",
            "data_distributor",
            "units",
            "updated_at",
        ]

    def get_collection_date(self, obj):
        if not obj.collection_date:
            return None
        collec_time = obj.last_collection_time or time.min
        dt = datetime.combine(obj.collection_date, collec_time)
        return timezone.make_aware(dt)

    def get_health_insight_categories(self, obj):
        return obj.health_insight.categories.values("id", "title")


class PatientInsightsDisplaySerializer(PatientHealthInsightSerializer):
    """ModelSerializer for Patient Health Insight with display attribute"""

    class Meta(PatientHealthInsightSerializer.Meta):
        fields = PatientHealthInsightSerializer.Meta.fields + ["display"]


class HealthDataChartSerializer(serializers.Serializer):
    """Serializer for Health Data Charting"""

    PERIODS = set(["day", "week", "month"])
    FREQS = {
        "day": "D",  # Daily
        "week": "W-MON",  # Monday is the start of the week
        "month": "MS",  # Month Start
        "year": "YS",  # Year Start
    }
    next = serializers.SerializerMethodField()
    previous = serializers.SerializerMethodField()
    external_patient_id = serializers.CharField()
    health_insight = serializers.SerializerMethodField()
    prev_to_date = serializers.SerializerMethodField()
    from_date = serializers.SerializerMethodField()
    to_date = serializers.SerializerMethodField()
    next_to_date = serializers.SerializerMethodField()
    timezone_info = serializers.SerializerMethodField()
    period = serializers.CharField()
    most_recent_collection_date = serializers.SerializerMethodField()
    chart_data = serializers.SerializerMethodField()
    chart_data_statistics = serializers.SerializerMethodField()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.ext_patient_id = self.initial_data["external_patient_id"]
            self.patient, _ = get_patient(self.ext_patient_id)
        except Exception as e:
            err_msg = f"Error querying for Ext. Patient `{self.ext_patient_id}`: {e}"
            raise exceptions.NotFound(err_msg) from e
        self.health_insight = self.initial_data["health_insight"]
        self.period = self.initial_data["period"]
        self.from_date = self.initial_data["from_date"]
        self.to_date = self.initial_data["to_date"]
        args = [self.patient, self.health_insight, self.from_date, self.to_date]
        self.period_avg = models.PatientHealthInsight.period_avg(
            *args, period=self.period
        ).exclude(avg__isnull=True)

    def fill_null_values(self, df, count):
        """Returns DataFrame with whole period range filled."""
        date_range = pd.date_range(
            end=self.to_date.date(), periods=count, freq=self.FREQS[self.period]
        )
        dt_format = {
            "day": "%d",
            "week": "%V",  # iso week format
            "month": "%m",
        }.get(self.period, "%Y-%m-%d")
        null_obj = {
            "year": date_range.strftime("%Y").astype(int),
            self.period: date_range.strftime(dt_format).astype(int),
            "avg": 0,
            "base_date": date_range.strftime("%Y-%m-%d"),
        }
        null_df = pd.DataFrame(null_obj, columns=df.columns, dtype="object")
        null_df.set_index(self.period, inplace=True, drop=False)
        df.set_index(self.period, inplace=True, drop=False)
        null_df.update(df)
        null_df[self.period] = null_df[self.period].astype("float").astype("Int64")
        null_df["year"] = null_df["year"].astype("float").astype("Int64")
        return null_df

    def get_chart_data(self, obj):
        """List of dictionaries with keys `period` and average of `value`"""
        columns = ["year", self.period, "avg", "base_date"]
        df = pd.DataFrame(list(self.period_avg), columns=columns)
        if not df.empty:
            df.avg = df.avg.astype(float)  # needed for rounding Decimal
            df.avg = df.avg.round(decimals=2)
            df = self.set_base_date(df)
        data_len = 7 if self.period == "day" else 6
        if len(df) < data_len:
            df = self.fill_null_values(df, data_len)
        df["base_date"] = pd.to_datetime(df["base_date"]).dt.date
        if self.period == "week" and self.from_date.year != self.to_date.year:
            # transition of year in week period apply weighted avg
            df["total_days"] = df.apply(
                lambda x: min(abs(x["base_date"] - date(self.from_date.year, 12, 31)).days, 7), axis=1
            )
            # update avg with its weighted value
            df["avg"] = df.apply(lambda df: df["avg"] * df["total_days"]/7, axis=1)
            agg_col_funcs = {'year': 'first', 'base_date': 'first', 'avg':  "sum"}
            # combine same week of different years
            df = df.groupby(df['week'], as_index=False).aggregate(agg_col_funcs)
            df.avg = df.avg.round(decimals=2)
        df = df.reindex(columns=columns)
        return df.sort_values(by="base_date", ascending=True).to_dict("records")

    def get_chart_data_statistics(self, obj):
        """Dictionary with keys `count`, `average`, `max`, `min`, `sum`
        calculated on `chart_data` result
        """
        chart_data = self.get_chart_data(obj)
        qs_aggregation = self.period_avg.aggregate(
            average=Avg("avg"), max=Max("avg"), min=Min("avg"), sum=Sum("avg")
        )
        return {
            "count": len(chart_data),
            "average": round(qs_aggregation["average"] or 0, 2),
            "max": round(qs_aggregation["max"] or 0, 2),
            "min": round(qs_aggregation["min"] or 0, 2),
            "sum": round(qs_aggregation["sum"] or 0, 2),
            "is_empty": self.period_avg.count() == 0,
        }

    def get_health_insight(self, *args):
        """Health Insight object selected properties"""
        return {
            "id": self.health_insight.id,
            "marker_name": self.health_insight.marker.name,
            "statistic_name": self.health_insight.statistic.name,
            "chart_type": self.health_insight.chart_type,
        }

    def get_most_recent_collection_date(self, *args):
        """Return collection datetime of the most recent Health Insight of the patient
        or None if no Health Insight exists
        """
        patient_insights = (
            self.patient.health_insights.filter(health_insight=self.health_insight)
            .exclude(value__isnull=True)
            .order_by("collection_date")
        )  # reverse of PatientHealthInsight
        if patient_insights.exists():
            return patient_insights.last().collection_date

    @classmethod
    def get_period_dates(cls, agg_type: str, to_date: date) -> tuple:
        """Return period for aggregation type based on base_date defaulting to today"""
        if agg_type not in cls.PERIODS:
            msg = f"Aggregation type `{agg_type}` not in {cls.PERIODS}"
            raise exceptions.ValidationError(msg)
        if not isinstance(to_date, date):
            try:
                to_date = dateparse.parse_date(to_date)
            except ValueError as e:
                raise exceptions.ValidationError(f"Invalid date `{to_date}` {e}") from e
        monday = to_date - timedelta(days=to_date.weekday())
        periods = {
            # Monday to Sunday
            "day": (monday, monday + timedelta(days=6)),
            # up to 6 weeks/months ago from first day of week/month
            "week": (monday - timedelta(weeks=5), to_date),  # Week starts on Monday
            "month": (to_date - relativedelta(months=5, day=1), to_date),
        }
        return periods[agg_type]

    def get_prev_to_date(self, *args):
        return (self.from_date - timedelta(days=1)).date()

    def get_from_date(self, *args):
        return self.from_date.date()

    def get_to_date(self, *args):
        return self.to_date.date()

    def get_next_to_date(self, *args):
        to_date = self.to_date.date()
        today = timezone.localdate()
        next_sunday = today + timedelta(days=6 - today.weekday())
        next_month = today + relativedelta(months=+1, day=1)
        if self.period == "day":
            days = 6 if self.to_date.weekday() < 6 else 7
            to_date = (self.to_date + relativedelta(days=+days)).date()
            if to_date > next_sunday:
                to_date = None
        elif self.period == "week":
            if to_date == next_sunday:
                to_date = None
            elif to_date > next_sunday:
                to_date = next_sunday
            else:
                to_date = (self.to_date + timedelta(weeks=+6)).date()
        elif self.period == "month":
           to_date = (self.to_date + relativedelta(months=+6, day=31)).date()
           if to_date >= next_month:
                to_date = None
        return to_date

    def get_timezone_info(self, *args):
        return str(self.to_date.tzinfo)

    def get_next(self, *args):
        to_date = self.get_next_to_date()
        return self.get_url(to_date)

    def get_previous(self, *args):
        to_date = self.get_prev_to_date()
        return self.get_url(to_date)

    def get_url(self, to_date: date):
        if to_date is None:
            return None
        params = {
            "ext_patient_id": self.ext_patient_id,
            "h_insight_id": self.health_insight.id,
        }
        request = self.context.get("request")
        url = request.build_absolute_uri(
            reverse("insights:patienthealthinsight-chart", kwargs=params)
        )
        tz = self.get_timezone_info()
        return f"{url}?period={self.period}&to_date={to_date}&timezone={tz}"

    def set_base_date(self, df):
        """Set base_date column in chart_data dataframe.
        Periods start on Monday or 1st if month.
        """
        if df.empty:
            return df
        begin = self.from_date
        end = self.to_date
        funcs = {
            # `day` sets base_date with Year and Day accounting for month turn
            "day": lambda df: date(df["year"], begin.month, df["day"]) if df["day"] >= begin.day else date(df["year"], end.month, df["day"]),
            # `week` sets base_date with Year and Week accounting for year turn
            "week": lambda df: date.fromisocalendar(df["year"], df["week"], 1) if df["year"] == begin.year else date.fromisocalendar(df["year"], 1, 1),
            # `month` sets base_date with Year and Month applying day 1 fixed
            "month": lambda df: date(df["year"], df["month"], 1),
        }
        df["base_date"] = (
            df[["year", self.period]].astype(int).apply(funcs[self.period], axis=1)
        )
        return df


class HealthInsightRecommendation(serializers.ModelSerializer):
    """ModelSerializer for Health Insight Recommendations"""

    status = serializers.CharField(source="get_status_display")
    from_insights = serializers.SerializerMethodField()

    class Meta:
        model = models.HealthInsightRecommendation
        fields = ["id", "title", "slug", "description", "status", "from_insights"]

    def get_from_insights(self, obj):
        return obj.health_insights.values(
            "id", marker_name=F("marker__name"), statistic_name=F("statistic__name")
        )


class PatientHealthInsightRecommendationSerializer(serializers.Serializer):
    """Serializer for Patient Health Insights Recommendations"""

    external_patient_id = serializers.CharField()
    recommendations = serializers.SerializerMethodField()

    class Meta:
        model = models.PatientHealthInsightRecommendation
        fields = ["external_patient_id", "recommendations"]

    def __init__(self, *args, **kwargs):
        # Default only active patient recommendations
        self.active_only = kwargs.pop("active_only", True)
        super().__init__(*args, **kwargs)

    def get_patient(self, obj):
        """Internal Patient object"""
        try:
            patient, _ = get_patient(obj["external_patient_id"])
        except:
            raise exceptions.NotFound(
                f"External Patient `{obj['external_patient_id']}` not found as Patient"
            )
        return patient

    def get_recommendations(self, obj):
        """List of dictionaries with keys `id`, `title`, `slug`, `description`,
        `status`, `from_insights`
        """
        return models.PatientHealthInsightRecommendation.get_for_patient(
            self.get_patient(obj), active_only=self.active_only
        )
