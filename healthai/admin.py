from django.contrib import admin
from .models import MoodEntry

@admin.register(MoodEntry)
class MoodEntryAdmin(admin.ModelAdmin):
    list_display = ('user', 'mood', 'date')   # Columns to display in admin list
    search_fields = ('user__username', 'mood') # Search by username and mood
    list_filter = ('mood', 'date')             # Filter by mood and date
