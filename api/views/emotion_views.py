from django.views.decorators.csrf import requires_csrf_token
from rest_framework import viewsets
from rest_framework import views, status, generics
from rest_framework.response import Response
from .prediction_views import Multimodal
from api.models import Emo_db
from api.serializers import Emo_dbSerializer

class EmotionCalendar(views.APIView):
    queryset = Emo_db.objects.all()
    serializer_class = Emo_dbSerializer

    def get(self, request, user_id):      
        date = request.GET.get('date', None)
        by_date = Emo_db.objects.filter(create_date=date)
        instance = generics.get_object_or_404(by_date, user_id=user_id)
        
        if str(instance.user_id) != user_id:
            return Response(data={"message" : "사용자 이름이 올바르지 않습니다."}, status=status.HTTP_400_BAD_REQUEST)
        
        queryset = Emo_db.objects.get(create_date=date)
        print(queryset)
        emo = queryset.emotion
                    
        return Response({'emotion' : emo}, status=status.HTTP_200_OK)