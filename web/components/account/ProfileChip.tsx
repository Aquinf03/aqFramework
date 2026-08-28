"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import { createClient } from "@/lib/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import Avvvatars from "avvvatars-react";
import {
  Camera,
  CircleNotch,
  Envelope,
  Check,
  SignOut,
  Key,
} from "@phosphor-icons/react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

interface Profile {
  name: string | null;
  avatar_url: string | null;
  email: string;
  waitlist_role: string | null;
  waitlist_company: string | null;
  waitlist_use_case: string | null;
}

function SettingRow({
  id,
  label,
  children,
}: {
  id: string;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div
      data-setting-id={id}
      className="grid grid-cols-[7.5rem_1fr] gap-x-6 items-center py-3.5 border-b border-stone-100 last:border-0"
    >
      <span className="text-sm text-stone-500">{label}</span>
      <div className="min-w-0">{children}</div>
    </div>
  );
}

export default function ProfileChip({ fullWidth, compact }: { fullWidth?: boolean; compact?: boolean }) {
  const { user } = useAuth();
  const [profile, setProfile] = useState<Profile | null>(null);
  const [open, setOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const supabase = useMemo(() => createClient(), []);

  const [tempName, setTempName] = useState("");
  const [savingName, setSavingName] = useState(false);
  const [uploadingAvatar, setUploadingAvatar] = useState(false);

  useEffect(() => {
    if (!user?.id) return;
    supabase
      .from("profiles")
      .select("name, avatar_url, email, waitlist_role, waitlist_company, waitlist_use_case")
      .eq("id", user.id)
      .single()
      .then(({ data, error }) => {
        if (data) {
          setProfile(data as Profile);
        } else {
          setProfile({
            name: null,
            avatar_url: null,
            email: user.email ?? "",
            waitlist_role: null,
            waitlist_company: null,
            waitlist_use_case: null,
          });
          if (error) console.warn("[ProfileChip] profile fetch:", error.message);
        }
      });
  }, [user?.id, supabase, user?.email]);

  useEffect(() => {
    if (profile) setTempName(profile.name || profile.email.split("@")[0]);
  }, [profile]);

  const handleSaveName = async () => {
    if (!tempName.trim() || !user?.id) return;
    setSavingName(true);
    const { error } = await supabase.from("profiles").update({ name: tempName.trim() }).eq("id", user.id);
    if (!error) setProfile(p => (p ? { ...p, name: tempName.trim() } : p));
    setSavingName(false);
  };

  const handlePasswordReset = async () => {
    if (!profile?.email) return;
    await supabase.auth.resetPasswordForEmail(profile.email, {
      redirectTo: `${window.location.origin}/auth/reset-password`,
    });
    setOpen(false);
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    window.location.href = "/";
  };

  const handleAvatarChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !user?.id) return;
    if (file.size > 5 * 1024 * 1024) return;
    setUploadingAvatar(true);
    try {
      const ext = file.name.split(".").pop();
      const filePath = `${user.id}/avatar.${ext}`;
      const { error: uploadError } = await supabase.storage.from("avatars").upload(filePath, file, { upsert: true });
      if (uploadError) throw uploadError;
      const { data: urlData } = supabase.storage.from("avatars").getPublicUrl(filePath);
      const publicUrl = `${urlData.publicUrl}?t=${Date.now()}`;
      await supabase.from("profiles").update({ avatar_url: publicUrl }).eq("id", user.id);
      setProfile(p => (p ? { ...p, avatar_url: publicUrl } : p));
    } catch (err) {
      console.error("[ProfileChip] avatar upload:", err);
    } finally {
      setUploadingAvatar(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  if (!user || !profile) return null;

  const displayName = profile.name || profile.email.split("@")[0];
  const avatarValue = profile.email || displayName;
  const hasAbout = !!(profile.waitlist_role || profile.waitlist_company || profile.waitlist_use_case);

  const avatarThumb = (size: number) =>
    profile.avatar_url ? (
      <img src={profile.avatar_url} alt="" className="size-full object-cover" />
    ) : (
      <Avvvatars value={avatarValue} style="shape" size={size} />
    );

  const inputClass =
    "w-full max-w-xs rounded-xl border border-stone-200 bg-stone-50 px-3 py-2 text-sm text-stone-800 outline-none focus:border-stone-300 focus:bg-white transition-colors";

  const stoneChipClass =
    "inline-flex items-center gap-2 rounded-xl bg-stone-300/40 hover:bg-stone-300/70 border border-stone-300/60 px-3 py-1.5 text-sm font-medium text-stone-700 transition-colors";

  const trigger = compact ? (
    <button
      type="button"
      onClick={() => setOpen(true)}
      className="size-9 rounded-full overflow-hidden shrink-0 ring-1 ring-stone-300/70 hover:ring-stone-400 transition-shadow"
      title={displayName}
    >
      {avatarThumb(36)}
    </button>
  ) : (
    <div
      className={cn(
        "flex items-center min-w-0 rounded-2xl border-2 border-black/10 overflow-hidden",
        fullWidth && "w-full",
      )}
    >
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="flex items-center gap-3 min-w-0 flex-1 px-3 py-2.5 hover:bg-stone-50 transition-colors text-left"
      >
        <div className="size-10 rounded-full overflow-hidden shrink-0">{avatarThumb(40)}</div>
        <div className="min-w-0 flex-1">
          <span className="block text-base font-medium text-stone-800 leading-tight truncate">{displayName}</span>
          <span className="block text-xs text-stone-400 leading-tight truncate mt-0.5">{profile.email}</span>
        </div>
      </button>
      <button
        type="button"
        onClick={() => void handleSignOut()}
        className="shrink-0 flex items-center gap-1.5 self-stretch px-3.5 border-l border-black/10 text-xs font-semibold text-stone-500 hover:text-red-700 hover:bg-red-50/80 transition-colors"
        title="Sign out"
      >
        <SignOut className="size-4" />
        <span className="hidden sm:inline">Sign out</span>
      </button>
    </div>
  );

  return (
    <>
      {trigger}

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent
          showCloseButton
          overlayClassName="bg-stone-900/15 backdrop-blur-[2px]"
          className="flex flex-col gap-0 p-0 overflow-hidden w-full max-w-md !rounded-3xl border-4 border-stone-400/60 shadow-2xl sm:max-w-md"
        >
          <DialogHeader className="px-6 pt-5 pb-1">
            <DialogTitle className="text-base font-semibold text-stone-900">Profile</DialogTitle>
            <DialogDescription className="text-xs text-stone-400">
              Manage your Aquin account
            </DialogDescription>
          </DialogHeader>

          <div className="px-6 pb-6 pt-3 overflow-y-auto no-scrollbar max-h-[min(70vh,560px)] space-y-4">
            <div className="rounded-2xl border border-stone-100 bg-stone-50/40 px-4">
              <SettingRow id="avatar" label="Avatar">
                <div className="flex items-center gap-3">
                  <div className="relative group shrink-0">
                    <div className="size-12 rounded-full overflow-hidden ring-1 ring-stone-200">
                      {avatarThumb(48)}
                    </div>
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={uploadingAvatar}
                      className="absolute inset-0 rounded-full bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity disabled:cursor-not-allowed"
                    >
                      {uploadingAvatar ? (
                        <CircleNotch className="size-4 text-white animate-spin" weight="bold" />
                      ) : (
                        <Camera className="size-4 text-white" weight="bold" />
                      )}
                    </button>
                  </div>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="text-sm text-stone-500 hover:text-stone-800 transition-colors"
                  >
                    Change photo
                  </button>
                </div>
              </SettingRow>

              <SettingRow id="name" label="Full name">
                <div className="flex items-center gap-2">
                  <input value={tempName} onChange={e => setTempName(e.target.value)} className={inputClass} />
                  <button
                    type="button"
                    onClick={() => void handleSaveName()}
                    disabled={savingName || tempName.trim() === (profile.name || displayName)}
                    className="shrink-0 p-2 rounded-xl bg-stone-300/40 hover:bg-stone-300/70 border border-stone-300/60 text-stone-500 hover:text-emerald-700 disabled:opacity-40 transition-colors"
                  >
                    {savingName ? (
                      <CircleNotch className="size-4 animate-spin" weight="bold" />
                    ) : (
                      <Check className="size-4" />
                    )}
                  </button>
                </div>
              </SettingRow>

              <SettingRow id="email" label="Email">
                <p className="text-sm text-stone-700">{profile.email}</p>
              </SettingRow>

              <SettingRow id="password" label="Password">
                <button type="button" onClick={() => void handlePasswordReset()} className={stoneChipClass}>
                  <Key className="size-4 text-stone-600" />
                  Send reset email
                </button>
              </SettingRow>

              <SettingRow id="session" label="Sign out">
                <button
                  type="button"
                  onClick={() => void handleSignOut()}
                  className={`${stoneChipClass} hover:bg-red-100/60 hover:border-red-200/80 hover:text-red-700`}
                >
                  <SignOut className="size-4 text-stone-600" />
                  Sign out
                </button>
              </SettingRow>

              <SettingRow id="contact" label="Contact">
                <a href="mailto:aquin@aquin.app" className={stoneChipClass}>
                  <Envelope className="size-4 text-stone-600" />
                  aquin@aquin.app
                </a>
              </SettingRow>
            </div>

            {hasAbout && (
              <div className="rounded-2xl border border-stone-100 bg-stone-50/40 px-4">
                {profile.waitlist_role && (
                  <SettingRow id="role" label="Role">
                    <p className="text-sm text-stone-700">{profile.waitlist_role}</p>
                  </SettingRow>
                )}
                {profile.waitlist_company && (
                  <SettingRow id="company" label="Company">
                    <p className="text-sm text-stone-700">{profile.waitlist_company}</p>
                  </SettingRow>
                )}
                {profile.waitlist_use_case && (
                  <SettingRow id="use-case" label="Use case">
                    <p className="text-sm text-stone-700 leading-relaxed">{profile.waitlist_use_case}</p>
                  </SettingRow>
                )}
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleAvatarChange} />
    </>
  );
}
