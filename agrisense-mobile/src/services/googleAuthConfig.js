import { GOOGLE_WEB_CLIENT_ID as WEB_ID, GOOGLE_IOS_CLIENT_ID as IOS_ID } from '@env';

export const GOOGLE_WEB_CLIENT_ID = WEB_ID;
export const GOOGLE_IOS_CLIENT_ID = IOS_ID;

export const isGoogleAuthConfigured = () => {
  return (
    !!GOOGLE_WEB_CLIENT_ID &&
    !GOOGLE_WEB_CLIENT_ID.startsWith('your-') &&
    !!GOOGLE_IOS_CLIENT_ID &&
    !GOOGLE_IOS_CLIENT_ID.startsWith('your-')
  );
};