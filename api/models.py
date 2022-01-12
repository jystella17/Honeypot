from enum import auto
import random
from re import U
from typing import Text
from django.db import models
from django.db.models import ImageField, FileField
from django.utils import timezone
from django.db.models.fields import *
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db.models.fields.related import ForeignKey


class UserManager(BaseUserManager):
    def create_user(self, username, nickname, email, **extra_fields):
        if not username:
            raise ValueError("The ID must be set.")

        self.normalize_email(email)
        username = email.strip().rsplit('@', 1)[0]
        user = self.model(username=username, nickname=nickname, email=email, **extra_fields)
        user.nickname = nickname
        extra_fields.setdefault('is_staff', False)
        extra_fields.setdefault('is_superuser', False)
        extra_fields.setdefault('last_login', timezone.now())
        extra_fields.setdefault('date_joined', timezone.now())

        user.save(using=self._db)
        return user

    def create_superuser(self, username, nickname='관리자', password='', **extra_fields):
        if not username:
            raise ValueError("The ID must be set.")

        user = self.model(username=username, nickname=nickname, password=password, **extra_fields)
        user.username = username
        user.nickname = nickname
        user.set_password(password)
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('last_login', timezone.now())
        extra_fields.setdefault('date_joined', timezone.now())
        if extra_fields.get('is_staff') is False:
            raise ValueError('Superuser must have is_staff = True')
        if extra_fields.get('is_superuser') is False:
            raise ValueError('Superuser must have is_superuser = True')

        user.save(using=self._db)
        return user
# ewhacyber17!

class User(AbstractBaseUser, PermissionsMixin):
    def __str__(self):
        return self.username

    USERNAME_FIELD = 'username'
    objects = UserManager()

    username = CharField(max_length=30, primary_key=True)
    email = EmailField(max_length=254, null=True)
    nickname = CharField(max_length=20)
    oauth_provider = CharField(max_length=20, null=True)
    access_token = CharField(max_length=254, null=True)
    refresh_token = CharField(max_length=254, null=True)
    exp_date = DateTimeField(null=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    last_login = models.DateTimeField(null=True, blank=True, auto_now_add=True)
    date_joined = models.DateTimeField(auto_now_add=True, null=True)

    class Meta:
        managed = True
        verbose_name = 'User'
        verbose_name_plural = 'Users'


class VoiceVector(models.Model):
    user_id = ForeignKey(to=User, on_delete=models.CASCADE, db_column='user_id')
    vector_voice = CharField(max_length=150)
    created_voice = DateField(auto_now_add=True)

    class Meta:
        managed = True
        verbose_name = 'voice_vector'
        verbose_name_plural = 'voice_vectors'
        

class TextVector(models.Model):
    user_id = ForeignKey(to=User, on_delete=models.CASCADE, db_column='user_id')
    vector_text = CharField(max_length=150)
    created = DateField(auto_now_add=True)
    
    class Meta:
        managed = True
        verbose_name = 'text_vector'
        verbose_name_plural = 'text_vectors'


class Emo_db(models.Model):
    user_id = ForeignKey(to=User, on_delete=models.CASCADE, db_column='user_id')
    voice_data = FileField(upload_to='voice-log/%Y/%m/%d'+'_'+str(random.randint(1000000,9999999)))
    create_date = DateField(auto_now_add=True)
    emotion = CharField(max_length=10)

    class Meta:
        managed = True
        verbose_name = 'emo_db'
        verbose_name_plural = 'emo_dbs'

