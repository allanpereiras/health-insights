import logging

from django.core.management.base import BaseCommand, CommandError
from insights.redshift import Redshift

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Add ID IDENTITY COLUMN to table provided as argument"

    def add_arguments(self, parser):
        def_table = "health_dev.test_health_standardized"
        parser.add_argument("--table", type=str, nargs="?", default=def_table)

    def query(self, table, command):
        temp_table = f"{table}_temp"
        new_name = table.split(".", 1)[1]
        table_creation = (
            f"DROP TABLE IF EXISTS {temp_table}; CREATE TABLE {temp_table} ("
            "id BIGINT IDENTITY(1,1), "
            "external_patient_id INTEGER ENCODE az64, "
            "distributor VARCHAR(50) ENCODE lzo, "
            "distributor_marker VARCHAR(50) ENCODE lzo, "
            "device_name VARCHAR(50) ENCODE lzo, "
            "device_version VARCHAR(50) ENCODE lzo, "
            "value_timestamp TIMESTAMP WITHOUT TIME ZONE ENCODE RAW, "
            "value_units VARCHAR(256) ENCODE lzo, "
            "distributor_version VARCHAR(256) ENCODE lzo, "
            "value DOUBLE PRECISION ENCODE RAW, "
            "ledger_uuid VARCHAR(40) ENCODE lzo, "
            "created_at TIMESTAMP WITHOUT TIME ZONE ENCODE az64, "
            "updated_at TIMESTAMP WITHOUT TIME ZONE ENCODE az64, "
            "value_timestamp_start TIMESTAMP WITHOUT TIME ZONE ENCODE az64, "
            "value_timestamp_end TIMESTAMP WITHOUT TIME ZONE ENCODE az64 ); "
        )  # only able to add IDENTITY column during table creation - using temp table
        cols = (
            "(external_patient_id, distributor, distributor_marker, device_name, "
            "device_version, value_timestamp, value_units, "
            "distributor_version, value, ledger_uuid, created_at, updated_at, "
            "value_timestamp_start, value_timestamp_end)"
        )
        copy_rows = (
            f"INSERT INTO {temp_table} {cols} "
            f"  SELECT * FROM {table} ORDER BY created_at ASC; "
        )
        rename = f"DROP TABLE {table}; ALTER TABLE {temp_table} RENAME TO {new_name}; "
        commands = {1: table_creation, 2: copy_rows, 3: rename}
        return commands[command]

    def handle(self, *args, **options):
        table = options["table"]
        try:
            for command in range(1, 4):
                query = self.query(table, command)
                with Redshift().get_cursor() as cursor:
                    cursor.execute(query)
            logger.info(f"--> Success: {table}")
        except Exception as e:
            msg = f"Error in query to add ID in table {table} {e}"
            logger.error(msg)
            raise CommandError(msg) from e
