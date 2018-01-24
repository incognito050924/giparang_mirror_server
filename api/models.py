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
    image = models.ImageField(upload_to='images/test/', blank=False)
    measured_at = models.DateTimeField(auto_now_add=True, editable=False)
    is_trained = models.BooleanField(default=False)
    score_pore = models.FloatField(blank=True, default=-1)
    score_wrinkle = models.FloatField(blank=True, default=-1)
    score_pigmentation = models.FloatField(blank=True, default=-1)
    score_erythema = models.FloatField(blank=True, default=-1)
    score_emotion = models.FloatField(blank=True, default=-1)
    score_total = models.FloatField(blank=True, default=-1)
    comment = models.TextField(default=None, blank=True, null=True)

    # class Meta:
    #     # DESC  : ['-colunm_name']
    #     # ASC   : [ 'colunm_name']
    #     ordering = ['-measured_at']

    def __str__(self):
        return '%s [ Pore: %.2f, Wrinkle: %.2f, Pigmentation: %.2f, Erythema: %.2f, Emotion: %.2f ]' \
               % (self.measured_at, self.score_pore, self.score_wrinkle, self.score_pigmentation, self.score_erythema, self.score_emotion)