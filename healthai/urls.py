# from django.urls import path
# from .views import SignupView, LoginView, LogoutView, log_mood, get_mood_history, chat_with_ai
# from rest_framework_simplejwt.views import TokenRefreshView

# urlpatterns = [
#     # ✅ Authentication Endpoints
#     path("signup/", SignupView.as_view(), name="signup"),
#     path("login/", LoginView.as_view(), name="login"),
#     path("logout/", LogoutView.as_view(), name="logout"),
#     path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),

#     # ✅ Mood Tracking Endpoints
#     path("mood/log/", log_mood, name="log-mood"),
#     path("mood/history/", get_mood_history, name="mood-history"),

#     # ✅ AI Chatbot Endpoint
#     path("chat/", chat_with_ai, name="chat_with_ai"),  # ✅ New chatbot route
# ]


from django.urls import path
from .views import (
    SignupView, LoginView, LogoutView, log_mood, get_mood_history,
    chat_with_ai, DashboardView
)
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    # ✅ Authentication Endpoints
    path("signup/", SignupView.as_view(), name="signup"),
    path("login/", LoginView.as_view(), name="login"),
    path("logout/", LogoutView.as_view(), name="logout"),
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),

    # ✅ Mood Tracking Endpoints
    path("mood/log/", log_mood, name="log-mood"),
    path("mood/history/", get_mood_history, name="mood-history"),

    # ✅ AI Chatbot Endpoint
    path("chat/", chat_with_ai, name="chat_with_ai"),

    # ✅ Dashboard Endpoint (Public Access)
    path("dashboard/", DashboardView.as_view(), name="dashboard"),
]
