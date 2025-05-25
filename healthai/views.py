# from healthai.ml_models.Mental_Health_Chatbot import chatbot_response
# from rest_framework.response import Response
# from rest_framework.views import APIView
# from rest_framework.decorators import api_view, permission_classes
# from rest_framework.permissions import AllowAny, IsAuthenticated
# from rest_framework_simplejwt.tokens import RefreshToken
# from django.contrib.auth.models import User
# from django.contrib.auth import authenticate
# from rest_framework import status
# from django.conf import settings
# from django.views.decorators.csrf import csrf_exempt
# from django.http import JsonResponse
# import requests
# import json
# import random


# from .models import MoodEntry
# from .serializers import MoodEntrySerializer

# # ✅ Signup View
# class SignupView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         try:
#             email = request.data.get("email")
#             username = request.data.get("username")
#             password = request.data.get("password")

#             if not email or not username or not password:
#                 return Response({"error": "All fields are required"}, status=status.HTTP_400_BAD_REQUEST)

#             if User.objects.filter(email=email).exists():
#                 return Response({"error": "Email already registered"}, status=status.HTTP_400_BAD_REQUEST)

#             user = User.objects.create_user(username=username, email=email, password=password)
#             refresh = RefreshToken.for_user(user)

#             return Response({
#                 "message": "User created successfully",
#                 "access": str(refresh.access_token),
#                 "refresh": str(refresh)
#             }, status=status.HTTP_201_CREATED)

#         except Exception as e:
#             print("Signup Error:", str(e))
#             return Response({"error": "Internal server error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# # ✅ Login View
# class LoginView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         try:
#             email = request.data.get("email")
#             password = request.data.get("password")

#             if not email or not password:
#                 return Response({"error": "Email and password are required"}, status=status.HTTP_400_BAD_REQUEST)

#             user = User.objects.get(email=email)
#             authenticated_user = authenticate(username=user.username, password=password)

#             if authenticated_user:
#                 refresh = RefreshToken.for_user(user)
#                 return Response({
#                     "access": str(refresh.access_token),
#                     "refresh": str(refresh)
#                 }, status=status.HTTP_200_OK)
#             else:
#                 return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)

#         except User.DoesNotExist:
#             return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

# # ✅ Logout View
# class LogoutView(APIView):
#     permission_classes = [IsAuthenticated]

#     def post(self, request):
#         try:
#             refresh_token = request.data.get("refresh")
#             if not refresh_token:
#                 return Response({"error": "Refresh token is required"}, status=status.HTTP_400_BAD_REQUEST)

#             token = RefreshToken(refresh_token)
#             token.blacklist()
#             return Response({"message": "Logged out successfully"}, status=status.HTTP_200_OK)

#         except Exception as e:
#             print("Logout Error:", str(e))
#             return Response({"error": "Invalid refresh token"}, status=status.HTTP_400_BAD_REQUEST)

# # ✅ Log Mood View
# @api_view(["POST"])
# @permission_classes([IsAuthenticated])
# def log_mood(request):
#     try:
#         mood = request.data.get("mood")
#         note = request.data.get("note", "")

#         if not mood:
#             return Response({"error": "Mood is required"}, status=status.HTTP_400_BAD_REQUEST)

#         mood_entry = MoodEntry.objects.create(user=request.user, mood=mood, note=note)
        
#         return Response({
#             "id": mood_entry.id,
#             "user": mood_entry.user.username,
#             "mood": mood_entry.mood,
#             "note": mood_entry.note,
#             "date": mood_entry.date
#         }, status=status.HTTP_201_CREATED)

#     except Exception as e:
#         print("Mood Logging Error:", str(e))
#         return Response({"error": "Failed to log mood"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# # ✅ Get Mood History View
# @api_view(["GET"])
# @permission_classes([IsAuthenticated])
# def get_mood_history(request):
#     try:
#         moods = MoodEntry.objects.filter(user=request.user).order_by("-date")
#         serializer = MoodEntrySerializer(moods, many=True)
#         return Response(serializer.data, status=status.HTTP_200_OK)

#     except Exception as e:
#         print("Mood History Error:", str(e))
#         return Response({"error": "Failed to fetch mood history"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# # ✅ Chatbot View (Now using your own trained model)
# @csrf_exempt
# @api_view(["POST"])
# @permission_classes([AllowAny])
# def chat_with_ai(request):
#     try:
#         data = json.loads(request.body)
#         user_message = data.get("message", "").strip()

#         if not user_message:
#             return JsonResponse({"error": "Message is required"}, status=400)

#         # ✅ Correct call to your trained model
#         ai_response = chatbot_response(user_message)

#         return JsonResponse({"response": ai_response}, status=200)

#     except Exception as e:
#         print("Chatbot Error:", str(e))
#         return JsonResponse({"error": "Chatbot failed to respond"}, status=500)

# # ✅ Dashboard View (Open for public access now)
# class DashboardView(APIView):
#     def get(self, request):
#         return Response({"message": "Welcome to the Dashboard!"}, status=status.HTTP_200_OK)



from healthai.ml_models.Mental_Health_Chatbot import chatbot_response
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework import status
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json

from .models import MoodEntry
from .serializers import MoodEntrySerializer

# ✅ Signup View
class SignupView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            email = request.data.get("email")
            username = request.data.get("username")
            password = request.data.get("password")

            if not email or not username or not password:
                return Response({"error": "All fields are required"}, status=status.HTTP_400_BAD_REQUEST)

            if User.objects.filter(email=email).exists():
                return Response({"error": "Email already registered"}, status=status.HTTP_400_BAD_REQUEST)

            user = User.objects.create_user(username=username, email=email, password=password)
            refresh = RefreshToken.for_user(user)

            return Response({
                "message": "User created successfully",
                "access": str(refresh.access_token),
                "refresh": str(refresh)
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            print("Signup Error:", str(e))
            return Response({"error": "Internal server error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ✅ Login View
class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            email = request.data.get("email")
            password = request.data.get("password")

            if not email or not password:
                return Response({"error": "Email and password are required"}, status=status.HTTP_400_BAD_REQUEST)

            user = User.objects.get(email=email)
            authenticated_user = authenticate(username=user.username, password=password)

            if authenticated_user:
                refresh = RefreshToken.for_user(user)
                return Response({
                    "access": str(refresh.access_token),
                    "refresh": str(refresh)
                }, status=status.HTTP_200_OK)
            else:
                return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)

        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

# ✅ Logout View
class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data.get("refresh")
            if not refresh_token:
                return Response({"error": "Refresh token is required"}, status=status.HTTP_400_BAD_REQUEST)

            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response({"message": "Logged out successfully"}, status=status.HTTP_200_OK)

        except Exception as e:
            print("Logout Error:", str(e))
            return Response({"error": "Invalid refresh token"}, status=status.HTTP_400_BAD_REQUEST)

# ✅ Log Mood View
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def log_mood(request):
    try:
        mood = request.data.get("mood")
        note = request.data.get("note", "")

        if not mood:
            return Response({"error": "Mood is required"}, status=status.HTTP_400_BAD_REQUEST)

        mood_entry = MoodEntry.objects.create(user=request.user, mood=mood, note=note)
        
        return Response({
            "id": mood_entry.id,
            "user": mood_entry.user.username,
            "mood": mood_entry.mood,
            "note": mood_entry.note,
            "date": mood_entry.date
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        print("Mood Logging Error:", str(e))
        return Response({"error": "Failed to log mood"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ✅ Get Mood History View
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_mood_history(request):
    try:
        moods = MoodEntry.objects.filter(user=request.user).order_by("-date")
        serializer = MoodEntrySerializer(moods, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    except Exception as e:
        print("Mood History Error:", str(e))
        return Response({"error": "Failed to fetch mood history"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ✅ Chatbot View (Now using your own trained model)
@csrf_exempt
@api_view(["POST"])
@permission_classes([AllowAny])
def chat_with_ai(request):
    try:
        data = json.loads(request.body)
        user_message = data.get("message", "").strip()

        if not user_message:
            return JsonResponse({"error": "Message is required"}, status=400)

        ai_response = chatbot_response(user_message)
        return JsonResponse({"response": ai_response}, status=200)

    except Exception as e:
        print("Chatbot Error:", str(e))
        return JsonResponse({"error": "Chatbot failed to respond"}, status=500)

# ✅ Dashboard View (Open to Public — No JWT Required)
class DashboardView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({"message": "Welcome to the Dashboard!"}, status=status.HTTP_200_OK)
