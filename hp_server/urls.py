from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include, re_path
from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from rest_framework import permissions
from api.views import login_views, prediction_views, emotion_views

schema_view = get_schema_view(
    openapi.Info(
        title="Swagger Study API",
        default_version="v1",
        description="Honeypot API 문서",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(name="test", email="test@test.com"),
        license=openapi.License(name="Test License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('allauth.urls')),
    path('accounts/google/login', login_views.GoogleLoginView.as_view(), name='google_login'),
    path('accounts/google/callback', login_views.GoogleCallbackView.as_view(), name='google_callback'),
    path('accounts/naver/login', login_views.NaverLoginView.as_view(), name='naver_login'),
    path('accounts/naver/login/callback', login_views.NaverCallbackView.as_view(), name='naver_callback'),
    path('accounts/kakao/login', login_views.KakaoLoginView.as_view(), name='kakao_login'),
    path('accounts/kakao/callback', login_views.KakaoCallbackView.as_view(), name='kakao_callback'),
    path('emotion/detection/prediction/voice', prediction_views.CNN.as_view(), name='voice_pred'),
    path('emotion/detection/prediction/text', prediction_views.KoBERT.as_view(), name='text_pred'),
    path('emotion/detection/prediction/multimodal', prediction_views.Multimodal.as_view(), name='multi_pred'),
    path('emotion/<str:user_id>/by-date', emotion_views.EmotionCalendar.as_view(), name='emotion_calendar'),
]

if settings.DEBUG:
    urlpatterns += [
        path('', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
        re_path(r'^swagger(?P<format>\.json|\.yaml)$', schema_view.without_ui(cache_timeout=0), name="schema-json"),
        re_path(r'^swagger/$', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
        re_path(r'^redoc/$', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    ]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)