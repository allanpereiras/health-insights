import logging

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from insights.models import (
    DataDevice,
    DeviceSupportedHealthInsight,
    HealthInsight,
    PatientDevice,
)
from insights.redshift import Redshift
from patients.utils import get_or_create_patient_with_external_id as get_patient

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Load data device from redshift"

    def add_arguments(self, parser):
        parser.add_argument(
            "--table", type=str, nargs="?", default="health_dev.health_standardized"
        )

    def handle(self, *args, **options):
        query = (
            "SELECT DISTINCT external_patient_id, distributor_marker, device_name, "
            f"device_version FROM {options['table']} "
            "WHERE external_patient_id IS NOT NULL AND device_name != '';"
        )
        try:
            results = Redshift().execute(query, [])
            for row in results:
                distrib_marker = row["distributor_marker"]
                params_data_device = {
                    "name": row["device_name"],
                    "version": row["device_version"],
                }
                data_device, _ = DataDevice.objects.get_or_create(**params_data_device)
                external_patient_id = row["external_patient_id"]
                patient, _ = get_patient(external_patient_id)
                params_patient_device = {
                    "patient": patient,
                    "device": data_device,
                }
                PatientDevice.objects.get_or_create(**params_patient_device)
                try:
                    health_insight = HealthInsight.objects.get(
                        marker__distributors__name=distrib_marker
                    )
                except HealthInsight.DoesNotExist:
                    continue  # next row
                params_device_sup_health_insight = {
                    "device": data_device,
                    "health_insight": health_insight,
                    "priority": 1,
                }
                DeviceSupportedHealthInsight.objects.get_or_create(
                    **params_device_sup_health_insight
                )
                logger.info("--> Success: %s", row)
        except Exception as e:
            logger.error(f"Error loading Devices data from Redshift: {e}")
            raise CommandError(f"Error loading Devices data from Redshift: {e}") from e
