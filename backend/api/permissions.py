from rest_framework.permissions import BasePermission

from .models import Profile


def get_user_role(user):
    if not user or not user.is_authenticated:
        return None
    if user.is_superuser or user.is_staff:
        return Profile.ROLE_ADMIN
    profile = getattr(user, "profile", None)
    return profile.role if profile else None

class IsAuthorizedUser(BasePermission):
    """
    Allows access only to users with is_authorized=True in their profile.
    """
    def has_permission(self, request, view):
        return (
            request.user and
            request.user.is_authenticated and
            hasattr(request.user, 'profile') and
            request.user.profile.is_authorized
        )


class IsDoctorOrAdmin(BasePermission):
    def has_permission(self, request, view):
        role = get_user_role(request.user)
        return role in {Profile.ROLE_ADMIN, Profile.ROLE_DOCTOR}


class IsAdminRole(BasePermission):
    def has_permission(self, request, view):
        return get_user_role(request.user) == Profile.ROLE_ADMIN
