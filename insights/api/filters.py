"""
Provides filtering backends that can be used to filter the results
returned by list views.

Filtering is done mostly using the package django_filters integrated with DRF.
"""

from django_filters import rest_framework as filters
from insights import models


class PatientHealthInsightFilter(filters.FilterSet):
    """
    Provides attributes for filtering results of Patient Health Insight.

    Attributes:
        attr:`~external_patient_id`
        attr:`~null_external_patient_id`
        attr:`~collection_date`
        attr:`~units`
    """

    external_patient_id = filters.CharFilter(
        field_name="patient__external_patients__external_id",
        label="External Patient Id",
    )
    start_date = filters.DateFilter(field_name="collection_date", lookup_expr="gte")
    end_date = filters.DateFilter(field_name="collection_date", lookup_expr="lte")
    data_distributor = filters.CharFilter(
        field_name="data_distributor__name",
        label="Data Distributor name",
    )
    marker_name = filters.CharFilter(
        field_name="health_insight__marker__name",
        label="Marker name",
    )

    class Meta:
        model = models.PatientHealthInsight
        fields = [
            "external_patient_id",
            "data_distributor",
            "start_date",
            "end_date",
            "units",
        ]


class DataDistributorLedgerFilter(filters.FilterSet):
    """
    Provides attributes for filtering results of Data Distributor Ledger.

    Attributes:
        attr:`~external_patient_id`
        attr:`~distributor`
        attr:`~date_time`
        attr:`~started_at`
        attr:`~finished_at`
        attr:`~is_processed`
        attr:`~processed_at`
        attr:`~file_location`
    """

    external_patient_id = filters.CharFilter(
        field_name="patient__external_patients__external_id",
        label="External Patient Id",
    )
    date_time = filters.IsoDateTimeFilter(
        method="get_between_start_finish", label="Date Time (in between start/finish)"
    )
    processed = filters.IsoDateTimeFromToRangeFilter(
        field_name="processed_at", label="Processed (Range)"
    )
    started = filters.IsoDateTimeFromToRangeFilter(
        field_name="started_at", label="Started (Range)"
    )
    finished = filters.IsoDateTimeFromToRangeFilter(
        field_name="finished_at", label="Finished (Range)"
    )

    def get_between_start_finish(self, queryset, field_name, value):
        return queryset.filter(started_at__lte=value, finished_at__gte=value)

    class Meta:
        model = models.DataDistributorLedger
        fields = [
            "external_patient_id",
            "distributor",
            "date_time",
            "started_at",
            "finished_at",
            "is_processed",
            "processed_at",
            "file_location",
        ]


class PatientRecommendationsFilter(filters.FilterSet):
    """
    Provides attributes for filtering results of Patient Recommendations.

    Attributes:
        attr:`~active_only`
    """

    active_only = filters.BooleanFilter(method="get_active", label="Is Active")

    class Meta:
        model = models.PatientHealthInsightRecommendation
        fields = ["active_only"]

    def get_active_only(self, queryset, field_name, value):
        if value:
            active = models.PatientHealthInsightRecommendation.ACTIVE_STATUS
            queryset = queryset.filter(status=active)
        return queryset
