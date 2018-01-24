from rest_framework import  serializers
from .models import User, SkinData
from .services import analysis

class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('user_id',)
        read_only_fields = ('user_id', 'created_at',)
    #
    # def create(self, validated_data):
    #     return User.objects.create(**validated_data)
    #
    # def update(self, instance, validated_data):
    #     instance.user_id = validated_data.get('user_id', instance.user_id)
    #     instance.created_at = validated_data.get('created_at', instance.created_at)
    #     instance.use_flag = validated_data.get('use_flag', instance.use_flag)
    #     instance.save()
    #     return instance


class SkinDataSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = SkinData
        fields = ('image', 'measured_at', 'is_trained',
                  'score_erythema', 'score_emotion', 'score_pigmentation', 'score_pore', 'score_wrinkle', 'score_total',
                  'comment',)

    # def create(self, validated_data):
    #     validated_data.update(analysis.get_score_data())
    #     return SkinData.objects.create(**validated_data)


class ResultSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = SkinData
        fields = ('measured_at', 'score_erythema', 'score_emotion', 'score_pigmentation', 'score_pore', 'score_wrinkle',
                  'score_total', 'comment',)

    # def create(self, validated_data):
    #     return SkinData.objects.create(**validated_data)

    # def update(self, instance, validated_data):
    #     instance.user = validated_data.get('user', instance.user)
    #     instance.image = validated_data.get('image', instance.image)
    #     instance.measured_at = validated_data('measured_at', instance.measured_at)
    #     instance.is_trained = validated_data('train_flag', instance.is_trained)
    #     instance.score_erythema = validated_data.get('score_erythema', instance.score_erythema)
    #     instance.score_emotion = validated_data.get('score_emotion', instance.score_emotion)
    #     instance.score_pigmentation = validated_data.get('score_pigmentation', instance.score_pigmentation)
    #     instance.score_pore = validated_data.get('score_pore', instance.score_pore)
    #     instance.score_wrinkle = validated_data.get('score_wrinkle', instance.score_wrinkle)
    #     instance.score_total = validated_data.get('score_total', instance.score_total)
    #     instance.comment = validated_data.get('comment', instance.comment)
    #     instance.save()
    #     return instance


class MeasuredSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = SkinData
        fields = ('image', 'measured_at', 'score_erythema', 'score_emotion', 'score_pigmentation', 'score_pore', 'score_wrinkle',
                  'score_total', 'comment',)

    # def create(self, validated_data):
    #     validated_data.update(analysis.get_score_data())
    #     return SkinData.objects.create(**validated_data)

