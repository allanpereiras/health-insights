# Generated by Django 3.1.5 on 2023-02-02 22:43

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('insights', '0035_initial_recommendationtriggers_20230110_2103'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='datadevice',
            unique_together={('name', 'version')},
        ),
    ]
