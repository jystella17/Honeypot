from json import JSONDecodeError
import requests
from urllib import parse

from django.contrib.auth import login
from django.http import JsonResponse
from django.conf import settings
from django.shortcuts import redirect
from rest_framework import status, generics
from rest_framework.response import Response

from api.models import User
from api.serializers import UserSerializer
from allauth.socialaccount.models import SocialAccount
from drf_yasg.utils import swagger_auto_schema

BASE_URL = 'http://localhost:8000/'
NAVER_CALLBACK_URI = 'http://localhost:8000/accounts/naver/callback'
KAKAO_CALLBACK_URI = 'http://localhost:8000/accounts/kakao/callback'
GOOGLE_CALLBACK_URI = 'http://localhost:8000/accounts/google/callback'
# GOOGLE_CALLBACK_URI = 'https://oauth.pstmn.io/v1/callback'


# Google Login Setting
class GoogleLoginView(generics.GenericAPIView):
    serializer_class = UserSerializer

    """
    Authorization Code Request
    """
    def post(self, request):
        scope = "https://www.googleapis.com/auth/userinfo.email"
        client_id = getattr(settings, "SOCIAL_AUTH_GOOGLE_CLIENT_ID")
        return redirect(
            f"https://accounts.google.com/o/oauth2/v2/auth?client_id={client_id}&response_type=code"
            f"&redirect_uri={GOOGLE_CALLBACK_URI}&scope={scope}"
        )


class GoogleCallbackView(generics.GenericAPIView):
    serializer_class = UserSerializer

    def post(self, request):
        grant_type = 'authorization_code'
        client_id = getattr(settings, "SOCIAL_AUTH_GOOGLE_CLIENT_ID")
        client_secret = getattr(settings, "SOCIAL_AUTH_GOOGLE_SECRET_KEY")
        code = request.GET.get('code')
        redirect_uri = GOOGLE_CALLBACK_URI
        state = request.GET.get('state')
        """
        Access Token Request
        """
        data = {'client_id': {client_id}, 'client_secret': {client_secret}, 'code': {code}, 'grant_type': {grant_type},
                'redirect_uri': {redirect_uri}, 'state': {state}}
        token = requests.post("https://oauth2.googleapis.com/token", data=data)

        token_json = token.json()
        error = token_json.get("error")
        if error is not None:
            raise JSONDecodeError(error)

        access_token = token_json.get('access_token')
        refresh_token = token_json.get('refresh_token')

        profile = requests.post(
            f"https://www.googleapis.com/oauth2/v1/tokeninfo?access_token={access_token}")

        profile_status = profile.status_code
        if profile_status != 200:
            return JsonResponse({'Error Message': 'failed to get profile'}, status=status.HTTP_400_BAD_REQUEST)

        profile_json = profile.json()
        nickname = profile_json['email']
        nickname = nickname.split("@")[0]
        email = profile_json['email']

        try:
            user = User.objects.get(nickname=nickname)
            if user is None:
                return JsonResponse({'Error Message': 'Not a social user'}, status=status.HTTP_400_BAD_REQUEST)
            if user.oauth_provider != 'google':
                return JsonResponse({'Error Message': 'Not a Google user'}, status=status.HTTP_400_BAD_REQUEST)

            else:
                login(request, user, backend="django.contrib.auth.backends.ModelBackend", )

        except User.DoesNotExist:
            new_user = User.objects.create(
                username=nickname,
                nickname=nickname,
                email=email,
                oauth_provider='google',
                access_token=access_token,
                refresh_token=refresh_token,
            )

            new_user.save()
            login(request, new_user, backend="django.contrib.auth.backends.ModelBackend", )

        return redirect(BASE_URL)


# Naver Login Setting
class NaverLoginView(generics.GenericAPIView):
    serializer_class = UserSerializer
    """
    Authorization Code Request
    """
    def post(self, request):
        client_id = getattr(settings, "SOCIAL_AUTH_NAVER_CLIENT_ID")
        state = request.GET.get('state')
        return redirect(f"https://nid.naver.com/oauth2.0/authorize?response_type=code"
                        f"&client_id={client_id}&redirect_uri={NAVER_CALLBACK_URI}&state={state}")


class NaverCallbackView(generics.GenericAPIView):
    serializer_class = UserSerializer

    def post(self, request, *args, **kwargs):
        grant_type = 'authorization_code'
        client_id = getattr(settings, "SOCIAL_AUTH_NAVER_CLIENT_ID")
        client_secret = getattr(settings, "SOCIAL_AUTH_NAVER_SECRET_KEY")
        code = request.GET.get('code')
        state = request.GET.get('state')
        """
        Access Token Request
        """
        data = {'grant_type': {grant_type}, 'client_id': {client_id}, 'client_secret': {client_secret},
                'code': {code}, 'state': {state}}

        token = requests.post("https://nid.naver.com/oauth2.0/token", data=data)

        token_json = token.json()
        error = token_json.get("error")

        if error is not None:
            raise JSONDecodeError(error)

        access_token = token_json.get("access_token")
        refresh_token = token_json.get("refresh_token")

        profile = requests.post(
            f"https://openapi.naver.com/v1/nid/me",
            headers={"Authorization": f"Bearer {access_token}"},)

        profile_status = profile.status_code
        if profile_status != 200:
            return JsonResponse({'Error Message': 'failed to get profile'}, status=status.HTTP_400_BAD_REQUEST)

        profile_json = profile.json()
        nickname = profile_json['response']['nickname']
        email = profile_json['response']['email']

        try:
            user = User.objects.get(nickname=nickname)
            if user is None:
                return JsonResponse({'Error Message': 'Not a social user'}, status=status.HTTP_400_BAD_REQUEST)
            if user.oauth_provider != 'naver':
                return JsonResponse({'Error Message': 'Not a Naver user'}, status=status.HTTP_400_BAD_REQUEST)

            else:
                login(request, user, backend="django.contrib.auth.backends.ModelBackend",)

        except User.DoesNotExist:
            new_user = User.objects.create(
                username=nickname,
                nickname=nickname,
                email=email,
                oauth_provider='naver',
                access_token=access_token,
                refresh_token=refresh_token,
            )

            new_user.save()
            login(request, new_user, backend="django.contrib.auth.backends.ModelBackend", )

        return redirect(BASE_URL)


# Kakao Login Setting
class KakaoLoginView(generics.GenericAPIView):
    serializer_class = UserSerializer

    def post(self, request):
        client_id = getattr(settings, "SOCIAL_AUTH_KAKAO_CLIENT_ID")
        return redirect(
            f"https://kauth.kakao.com/oauth/authorize?client_id={client_id}&redirect_uri={KAKAO_CALLBACK_URI}"
            f"&response_type=code"
        )


class KakaoCallbackView(generics.GenericAPIView):
    serializer_class = UserSerializer

    def post(self, request):
        grant_type = 'authorization_code'
        client_id = getattr(settings, "SOCIAL_AUTH_KAKAO_CLIENT_ID")
        code = request.GET.get("code")
        redirect_uri = KAKAO_CALLBACK_URI
        """
        Access Token Request
        """
        data = {'grant_type': {grant_type}, 'client_id': {client_id}, 'redirect_uri': {redirect_uri}, 'code': {code}}
        token = requests.post(
            f"https://kauth.kakao.com/oauth/token", data=data)

        token_json = token.json()
        error = token_json.get("error")

        if error is not None:
            raise JSONDecodeError(error)
        access_token = token_json.get("access_token")
        print(access_token)
        refresh_token = token_json.get("refresh_token")

        headers = {'Authorization': f'Bearer {access_token}'}
        profile = requests.post(
            "https://kapi.kakao.com/v2/user/me", headers=headers)
        profile_json = profile.json()
        error = profile_json.get("error")

        if error is not None:
            raise JSONDecodeError(error)

        nickname = profile_json['properties']['nickname']
        email = profile_json['kakao_account']['email']

        try:
            user = User.objects.get(nickname=nickname)
            if user is None:
                return JsonResponse({'Error Message': 'Not a social user'}, status=status.HTTP_400_BAD_REQUEST)
            if user.oauth_provider != 'kakao':
                return JsonResponse({'Error Message': 'Not a Kakao user'}, status=status.HTTP_400_BAD_REQUEST)

            else:
                login(request, user, backend="django.contrib.auth.backends.ModelBackend",)

        except User.DoesNotExist:
            new_user = User.objects.create(
                username=nickname,
                nickname=nickname,
                email=email,
                oauth_provider='kakao',
                access_token=access_token,
                refresh_token=refresh_token,
            )
            new_user.save()
            login(request, new_user, backend="django.contrib.auth.backends.ModelBackend",)

        return redirect(BASE_URL)
