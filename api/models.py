from django.db import models


# Model(Domain) declarations.

class User(models.Model):
    """

    """
    user_id = models.CharField(max_length=100)
    # gender = models.IntegerField(max_length=1)
    # birth_date = models.DateField(default=None)
    use_flag = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True, editable=False)

    def __str__(self):
        return self.user_id


class SkinData(models.Model):
    """

    """
    # user = models.ForeignKey(User)
    # Measured data meta
    image = models.ImageField(upload_to='images/test/', blank=False)
    measured_at = models.DateTimeField(auto_now_add=True, editable=False)
    is_trained = models.BooleanField(default=False)
    # Pore data { score, num }
    score_pore = models.FloatField(blank=True, default=-1)
    pore_num = models.IntegerField(blank=True, default=0)
    # Wrinkle data { score, num, avarge_area, darkness, pitch, length }
    score_wrinkle = models.FloatField(blank=True, default=-1)
    wrinkle_num = models.IntegerField(blank=True, default=0)
    wrinkle_average_area = models.IntegerField(blank=True, default=0)
    wrinkle_darkness = models.IntegerField(blank=True, default=0)
    wrinkle_pitch = models.IntegerField(blank=True, default=0)
    wrinkle_length = models.IntegerField(blank=True, default=0)
    # Pigmentation data { score, num, avarge_area, darkness }
    score_pigmentation = models.FloatField(blank=True, default=-1)
    pigmentation_num = models.IntegerField(blank=True, default=0)
    pigmentation_average_area = models.IntegerField(blank=True, default=0)
    pigmentation_darkness = models.IntegerField(blank=True, default=0)
    # Erythema data { score, num, avarge_area, darkness }
    score_erythema = models.FloatField(blank=True, default=-1)
    erythema_num = models.IntegerField(blank=True, default=0)
    erythema_average_area = models.IntegerField(blank=True, default=0)
    erythema_darkness = models.IntegerField(blank=True, default=0)
    # Emotion data
    score_emotion = models.FloatField(blank=True, default=-1)
    emotion = models.CharField(blank=True, null=True, max_length=30)
    # Total data
    score_total = models.FloatField(blank=True, default=-1)
    comment = models.TextField(default=None, blank=True, null=True)

    # class Meta:
    #     # DESC  : ['-colunm_name']
    #     # ASC   : [ 'colunm_name']
    #     ordering = ['-measured_at']

    def __str__(self):
        return '%s [ Pore: %.2f, Wrinkle: %.2f, Pigmentation: %.2f, Erythema: %.2f, Emotion: %.2f ]' \
               % (self.measured_at, self.score_pore, self.score_wrinkle, self.score_pigmentation, self.score_erythema, self.score_emotion)