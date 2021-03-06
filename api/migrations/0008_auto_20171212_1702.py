# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2017-12-12 08:02
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0007_auto_20171212_1606'),
    ]

    operations = [
        migrations.AlterField(
            model_name='skindata',
            name='image',
            field=models.ImageField(editable=False, upload_to='images/test/'),
        ),
        migrations.AlterField(
            model_name='skindata',
            name='score_emotion',
            field=models.FloatField(blank=True),
        ),
        migrations.AlterField(
            model_name='skindata',
            name='score_erythema',
            field=models.FloatField(blank=True),
        ),
        migrations.AlterField(
            model_name='skindata',
            name='score_pigmentation',
            field=models.FloatField(blank=True),
        ),
        migrations.AlterField(
            model_name='skindata',
            name='score_pore',
            field=models.FloatField(blank=True),
        ),
        migrations.AlterField(
            model_name='skindata',
            name='score_total',
            field=models.FloatField(blank=True),
        ),
        migrations.AlterField(
            model_name='skindata',
            name='score_wrinkle',
            field=models.FloatField(blank=True),
        ),
    ]
