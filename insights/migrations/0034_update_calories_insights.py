# Generated by Django 3.1.5 on 2022-12-19 18:50
import logging

from django.conf import settings
from django.db import migrations

logger = logging.getLogger(__name__)


class Migration(migrations.Migration):
    def update_insights_calories_types(self, schema_editor):
        if settings.TESTING:
            return
        logger.setLevel(logging.INFO)
        Marker = self.get_model("insights", "Marker")
        DataDistributor = self.get_model("insights", "DataDistributor")
        DistributorMarker = self.get_model("insights", "DistributorMarker")
        DistributorMarkerPair = self.get_model("insights", "DistributorMarkerPair")
        HealthInsight = self.get_model("insights", "HealthInsight")
        PatientHealthInsight = self.get_model("insights", "PatientHealthInsight")

        try:
            resting_calories, _ = Marker.objects.get_or_create(name="Calories Burned at Rest")
            logger.info(f"Created new Marker `{resting_calories.name}`")
            total_calories, _ = Marker.objects.get_or_create(name="Total Calories Burned")
            logger.info(f"Created new Marker `{total_calories.name}`")
            DistributorMarker.objects.filter(
                name="com.google.calories.expended"
            ).update(marker=total_calories)
            logger.info(
                "Updated DistributorMarker name of `com.google.calories.expended`"
                f" to `{total_calories.name}`"
            )
            google, _ = DataDistributor.objects.get_or_create(
                name="GOOGLE_FIT", defaults={"base_date": "2014-10-28"}
            )
            bmr, _ = DistributorMarker.objects.get_or_create(
                name="com.google.calories.bmr",
                marker=resting_calories,
                distributor=google,
            )
            logger.info(
                f"Created new DistributorMarker for `{resting_calories.name}` "
                f"and `{google.name}`"
            )

            active_calories, _ = Marker.objects.get_or_create(name="Calories Burned")
            insight_active = HealthInsight.objects.get(marker=active_calories)
            # Create new Health Insight object for Resting Calories
            logger.info(f"Creating new HealthInsight `{active_calories}`")
            insight_resting, _ = HealthInsight.objects.get_or_create(
                marker=resting_calories,
                statistic=insight_active.statistic,
                default_duration=insight_active.default_duration,
                start_from=insight_active.start_from,
                backwards=insight_active.backwards,
                in_devices_aggregation=insight_active.in_devices_aggregation,
                chart_type=insight_active.chart_type,
            )
            # Create new Health Insight object for Total Calories
            logger.info(f"Creating new HealthInsight `{total_calories}`")
            insight_total, _ = HealthInsight.objects.get_or_create(
                marker=total_calories,
                statistic=insight_active.statistic,
                default_duration=insight_active.default_duration,
                start_from=insight_active.start_from,
                backwards=insight_active.backwards,
                in_devices_aggregation=insight_active.in_devices_aggregation,
                chart_type=insight_active.chart_type,
            )

            logger.info(
                "Creating new DistributorMarkerPair for pairing "
                "`com.google.calories.expended` and `com.google.calories.bmr`"
            )
            DistributorMarkerPair.objects.get_or_create(
                marker=active_calories,
                distributor=google,
                name="com.google.calories.active.derived",
                first=DistributorMarker.objects.get(
                    name="com.google.calories.expended"
                ),
                value_function="subtract",
                second=bmr,
                units="calories",
            )

            n_rows = PatientHealthInsight.objects.filter(
                health_insight__marker__name="Calories Burned",
                data_distributor=google,
            ).update(health_insight=insight_total)
            logger.info(
                f"Updated {n_rows} PatientHealthInsight `Calories Burned` "
                f"from `GOOGLE_FIT` to `{insight_total.marker.name}` new HealthInsight"
            )
        except Exception as e:
            logger.error(f"Failed performing data migration 0034: {e}")
            return

    dependencies = [
        ("insights", "0033_distributormarkerpair"),
    ]

    operations = [
        migrations.RunPython(update_insights_calories_types, migrations.RunPython.noop),
    ]
