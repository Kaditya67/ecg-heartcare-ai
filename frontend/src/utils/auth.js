export const getStoredUser = () => {
  try {
    const raw = localStorage.getItem('authUser');
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
};

export const getStoredRole = () => {
  const explicitRole = localStorage.getItem('authRole');
  if (explicitRole) return explicitRole;
  return getStoredUser()?.profile?.role ?? null;
};

export const storeAuthSession = (payload) => {
  localStorage.setItem('accessToken', payload.access);
  localStorage.setItem('refreshToken', payload.refresh);
  localStorage.setItem('authUser', JSON.stringify(payload.user));
  localStorage.setItem('authRole', payload.role || payload.user?.profile?.role || '');
};

export const clearAuthSession = () => {
  localStorage.removeItem('accessToken');
  localStorage.removeItem('refreshToken');
  localStorage.removeItem('authUser');
  localStorage.removeItem('authRole');
};

export const updateStoredUserProfile = (profile) => {
  const user = getStoredUser();
  if (!user) return;
  const updatedUser = { ...user, profile: { ...user.profile, ...profile } };
  localStorage.setItem('authUser', JSON.stringify(updatedUser));
};
