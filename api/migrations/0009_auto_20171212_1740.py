# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2017-12-12 08:40
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0008_auto_20171212_1702'),
    ]

    operations = [
        migrations.AlterField(
            model_name='skindata',
            name='image',
            field=models.ImageField(upload_to='images/test/'),
        ),
    ]
