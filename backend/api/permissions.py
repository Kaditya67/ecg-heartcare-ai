from rest_framework.permissions import BasePermission

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
