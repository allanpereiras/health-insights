"""This file is used for registering the models into the Django Admin interface"""

from django.contrib import admin
from insights import models


@admin.register(models.DataDistributorLedger)
class DataDistributorLedgerAdmin(admin.ModelAdmin):
    ordering = ["-created_at"]
    autocomplete_fields = ["patient"]
    search_fields = ["uuid", "distributor__name", "patient"]
    list_filter = ["distributor", "is_processed"]
    list_display = [
        "uuid",
        "patient",
        "distributor",
        "started_at",
        "finished_at",
        "is_processed",
        "processed_at",
        "file_location",
        "created_at",
        "updated_at",
    ]

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(models.DataDevice)
class DataDeviceAdmin(admin.ModelAdmin):
    list_display = ["name", "version", "created_at", "updated_at"]


@admin.register(models.PatientDevice)
class PatientDeviceAdmin(admin.ModelAdmin):
    list_display = ["patient", "external_patient", "device", "created_at", "updated_at"]

    def external_patient(self, obj):
        try:
            external_patients = obj.patient.external_patients
            return list(
                external_patients.values_list("external_id", flat=True).distinct()
            )
        except AttributeError:
            return

    external_patient.short_description = "External Patient ID(s)"


@admin.register(models.DeviceSupportedHealthInsight)
class DeviceSupportedHealthInsightAdmin(admin.ModelAdmin):
    list_display = ["device", "health_insight", "created_at", "updated_at"]
    list_filter = ["health_insight"]


@admin.register(models.HealthInsight)
class HealthInsightAdmin(admin.ModelAdmin):
    ordering = ["-created_at"]
    search_fields = ["id", "marker__name"]
    list_display = [
        "id",
        "marker",
        "statistic",
        "default_duration",
        "start_from",
        "backwards",
        "in_devices_aggregation",
        "chart_type",
        "created_at",
        "updated_at",
    ]


@admin.register(models.HealthInsightStateValue)
class HealthInsightStateValueAdmin(admin.ModelAdmin):
    ordering = ["-created_at"]
    autocomplete_fields = ["health_insight"]
    list_filter = ["health_insight", "applies_to_gender"]
    list_display = [
        "id",
        "health_insight",
        "applies_to_gender",
        "state",
        "upper_limit",
        "lower_limit",
        "reversed_cumulative",
        "created_at",
        "updated_at",
    ]


@admin.register(models.HealthInsightCategory)
class HealthInsightCategoryAdmin(admin.ModelAdmin):
    ordering = ["-created"]
    search_fields = ["id", "title", "description"]
    autocomplete_fields = ["health_insights"]
    list_filter = ["health_insights"]
    list_display = [
        "id",
        "health_insight_ids",
        "title",
        "slug",
        "description",
        "created",
        "modified",
    ]

    def health_insight_ids(self, obj):
        return list(obj.health_insights.all().values_list("id", flat=True))


@admin.register(models.Marker)
class MarkerAdmin(admin.ModelAdmin):
    ordering = ["-created_at"]
    search_fields = ["id", "name"]
    list_display = ["id", "name", "created_at", "updated_at"]


@admin.register(models.DataDistributor)
class DataDistributorAdmin(admin.ModelAdmin):
    ordering = ["-created_at"]
    search_fields = ["id", "name", "base_date"]
    list_display = ["id", "name", "base_date", "created_at", "updated_at"]


@admin.register(models.DistributorMarker)
class DistributorMarkerAdmin(admin.ModelAdmin):
    ordering = ["-created_at"]
    search_fields = ["id", "marker__name", "name"]
    list_display = ["id", "marker", "distributor", "name", "created_at", "updated_at"]


@admin.register(models.DistributorMarkerPair)
class DistributorMarkerPairAdmin(admin.ModelAdmin):
    ordering = ["-created_at"]
    search_fields = ["id", "marker__name", "name"]
    list_display = [
        "id",
        "marker",
        "distributor",
        "name",
        "first",
        "value_function",
        "second",
        "units",
        "created_at",
        "updated_at",
    ]


@admin.register(models.PatientHealthInsight)
class PatientHealthInsightAdmin(admin.ModelAdmin):
    ordering = ["-created_at"]
    autocomplete_fields = ["patient"]
    list_display = [
        "health_insight",
        "patient",
        "external_patient",
        "data_distributor",
        "collection_date",
        "last_collection_time",
        "value",
        "display",
        "created_at",
        "updated_at",
    ]
    list_filter = ["health_insight", "data_distributor"]
    search_fields = [
        "value",
        "collection_date",
        "patient__external_patients__external_id",
    ]

    def external_patient(self, obj):
        try:
            external_patients = obj.patient.external_patients
            return list(
                external_patients.values_list("external_id", flat=True).distinct()
            )
        except AttributeError:
            return

    external_patient.short_description = "External Patient ID(s)"


@admin.register(models.HealthInsightDisplayRule)
class HealthInsightDisplayRuleAdmin(admin.ModelAdmin):
    ordering = ["created_at"]
    list_display = [
        "id",
        "rule_set",
        "field",
        "comparator",
        "comparison_value",
        "comparison_value_type",
        "created_at",
        "updated_at",
    ]


@admin.register(models.HealthInsightDisplayRuleSet)
class HealthInsightDisplayRuleSetAdmin(admin.ModelAdmin):
    ordering = ["created_at"]
    list_display = [
        "id",
        "health_insight",
        "priority",
        "name",
        "created_at",
        "updated_at",
    ]


@admin.register(models.Goal)
class GoalAdmin(admin.ModelAdmin):
    ordering = ["-created"]
    search_fields = ["id", "name"]
    list_display = ["id", "name", "aggregation_type", "aggregation_period", "value", "value_unit", "created", "modified"]


@admin.register(models.HealthInsightRecommendation)
class HealthInsightRecommendationAdmin(admin.ModelAdmin):
    ordering = ["-created"]
    autocomplete_fields = ["health_insights"]
    search_fields = ["title", "description"]
    list_filter = ["health_insights", "status"]
    list_display = [
        "id",
        "health_insight_ids",
        "status",
        "title",
        "slug",
        "description",
        "created",
        "modified",
    ]

    def health_insight_ids(self, obj):
        return list(obj.health_insights.all().values_list("id", flat=True))


@admin.register(models.RecommendationTrigger)
class RecommendationTriggerAdmin(admin.ModelAdmin):
    ordering = ["-created"]
    search_fields = ["id", "title", "description"]
    list_display = ["id", "recommendation", "title", "description", "slug", "created", "modified"]


@admin.register(models.PatientHealthInsightRecommendation)
class PatientHealthInsightRecommendationAdmin(admin.ModelAdmin):
    ordering = ["-created"]
    autocomplete_fields = ["patient"]
    search_fields = [
        "patient__id",
        "recommendation__title",
        "recommendation__description",
    ]
    list_filter = ["status", "recommendation__health_insights", "recommendation"]
    list_display = [
        "id",
        "patient",
        "external_patient",
        "health_insights",
        "status",
        "recommendation",
        "created",
        "modified",
    ]

    def health_insights(self, obj):
        return list(
            obj.recommendation.health_insights.all().values_list("id", flat=True)
        )

    def external_patient(self, obj):
        try:
            external_patients = obj.patient.external_patients
            return list(
                external_patients.values_list("external_id", flat=True).distinct()
            )
        except AttributeError:
            return

    external_patient.short_description = "External Patient ID(s)"
