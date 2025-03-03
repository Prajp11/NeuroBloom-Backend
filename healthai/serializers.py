from rest_framework import serializers
from .models import MoodEntry

class MoodEntrySerializer(serializers.ModelSerializer):
    class Meta:
        model = MoodEntry
        fields = ["id", "user", "mood", "note", "date"]
        read_only_fields = ["user", "date"] 
