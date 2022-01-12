from django.contrib import admin
from .models import TextVector, User, Emo_db, VoiceVector


admin.site.register(User)
admin.site.register(Emo_db)
admin.site.register(TextVector)
admin.site.register(VoiceVector)
# Register your models here.
