from django.core.management.base import BaseCommand, CommandError
from insights.tests import factories


class Command(BaseCommand):
    help = "Creates initial seed data for Insights App"

    def handle(self, *args, **options):
        for Factory, count in factories.get_seeding_list():
            try:
                for _ in range(count):
                    seed = Factory()
                    self.stdout.write(
                        self.style.SUCCESS(
                            'Created "%s" Seed from "%s"' % (seed, Factory.__name__)
                        )
                    )
            except Exception as e:
                raise CommandError(f"An error ocurred while Seeding data: {e}") from e
