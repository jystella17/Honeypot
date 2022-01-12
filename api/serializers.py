from django.db.models import fields
from rest_framework import serializers
from .models import TextVector, User, Emo_db, VoiceVector


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'


class VoiceVectorSerializer(serializers.ModelSerializer):
    class Meta:
        model = VoiceVector
        fields = '__all__'
        
        
class TextVectorSerializer(serializers.ModelSerializer):
    class Meta:
        model = TextVector
        fields = '__all__'


class Emo_dbSerializer(serializers.ModelSerializer):
    class Meta:
        model = Emo_db
        fields = '__all__'
