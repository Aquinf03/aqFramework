"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import { createClient } from "@/lib/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import Avvvatars from "avvvatars-react";
import { CircleNotch } from "@phosphor-icons/react";
import {
  Popover,
  PopoverContent,
  PopoverDescription,
  PopoverHeader,
  PopoverTitle,
  PopoverTrigger,
} from "@/components/ui/popover";

interface Profile {
  name: string | null;
  avatar_url: string | null;
  email: string;
}

export default function ProfileChip() {
  const { user } = useAuth();
  const [profile, setProfile] = useState<Profile | null>(null);
  const [open, setOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const supabase = useMemo(() => createClient(), []);

  const [tempName, setTempName] = useState("");
  const [savingName, setSavingName] = useState(false);
  const [uploadingAvatar, setUploadingAvatar] = useState(false);
  const [avatarFileName, setAvatarFileName] = useState<string | null>(null);

  useEffect(() => {
    if (!user?.id) return;
    supabase
      .from("profiles")
      .select("name, avatar_url, email")
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
          });
          if (error) console.warn("[ProfileChip] profile fetch:", error.message);
        }
      });
  }, [user?.id, supabase, user?.email]);

  useEffect(() => {
    if (profile) setTempName(profile.name || profile.email.split("@")[0]);
  }, [profile]);

  useEffect(() => {
    if (!profile?.avatar_url) {
      setAvatarFileName(null);
      return;
    }
    try {
      const path = new URL(profile.avatar_url).pathname;
      const saved = path.split("/").pop();
      setAvatarFileName(saved && saved !== "" ? saved : "avatar");
    } catch {
      setAvatarFileName("avatar");
    }
  }, [profile?.avatar_url]);

  const handleSaveName = async () => {
    if (!tempName.trim() || !user?.id) return;
    const next = tempName.trim();
    if (next === (profile?.name || profile?.email.split("@")[0])) return;
    setSavingName(true);
    const { error } = await supabase.from("profiles").update({ name: next }).eq("id", user.id);
    if (!error) setProfile(p => (p ? { ...p, name: next } : p));
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
      setAvatarFileName(file.name);
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

  const avatarThumb = (size: number) =>
    profile.avatar_url ? (
      <img src={profile.avatar_url} alt="" className="size-full object-cover" />
    ) : (
      <Avvvatars value={avatarValue} style="shape" size={size} />
    );

  const linkTextClass =
    "text-sm text-stone-600 underline underline-offset-[3px] decoration-stone-300 transition-colors hover:text-stone-900 hover:decoration-stone-500";

  const nameInputClass =
    "w-full min-w-0 border-0 bg-transparent p-0 text-sm text-stone-600 underline underline-offset-[3px] decoration-stone-300 outline-none transition-colors placeholder:text-stone-400 focus:text-stone-900 focus:decoration-stone-500";

  return (
    <>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <button
            type="button"
            className="size-9 shrink-0 overflow-hidden rounded-full ring-1 ring-stone-300/70 outline-none transition-shadow hover:ring-stone-400 data-[state=open]:ring-stone-500"
            title={displayName}
          >
            {avatarThumb(36)}
          </button>
        </PopoverTrigger>
        <PopoverContent
          align="end"
          side="bottom"
          sideOffset={8}
          className="w-[min(calc(100vw-2rem),20rem)] rounded-2xl border-stone-200 bg-white p-0 shadow-xl"
        >
          <PopoverHeader className="gap-0.5 px-5 pt-4 pb-2">
            <PopoverTitle className="text-base font-semibold text-stone-900 font-host-grotesk tracking-[-0.02em]">
              Profile
            </PopoverTitle>
            <PopoverDescription className="text-xs text-stone-400">
              Manage your Aquin account
            </PopoverDescription>
          </PopoverHeader>

          <div className="max-h-[min(70vh,560px)] overflow-y-auto px-5 pb-5 pt-2 no-scrollbar">
            <div className="flex flex-col items-start gap-2.5">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploadingAvatar}
                className={`${linkTextClass} max-w-full truncate text-left disabled:opacity-50`}
              >
                {uploadingAvatar ? (
                  <span className="inline-flex items-center gap-2">
                    <CircleNotch className="size-3.5 animate-spin" weight="bold" />
                    Uploading…
                  </span>
                ) : avatarFileName ? (
                  avatarFileName
                ) : (
                  "Change photo"
                )}
              </button>
              <div className="relative w-full">
                <input
                  value={tempName}
                  onChange={e => setTempName(e.target.value)}
                  onBlur={() => void handleSaveName()}
                  onKeyDown={e => {
                    if (e.key === "Enter") {
                      e.currentTarget.blur();
                    }
                  }}
                  disabled={savingName}
                  aria-label="Full name"
                  className={nameInputClass}
                />
                {savingName ? (
                  <CircleNotch
                    className="absolute right-0 top-1/2 size-3.5 -translate-y-1/2 animate-spin text-stone-400"
                    weight="bold"
                  />
                ) : null}
              </div>

              <p className={linkTextClass}>{profile.email}</p>

              <button type="button" onClick={() => void handlePasswordReset()} className={linkTextClass}>
                Send reset email
              </button>
              <button
                type="button"
                onClick={() => void handleSignOut()}
                className={`${linkTextClass} hover:text-red-700 hover:decoration-red-300`}
              >
                Sign out
              </button>
              <a href="mailto:aquin@aquin.app" className={linkTextClass}>
                aquin@aquin.app
              </a>
            </div>
          </div>
        </PopoverContent>
      </Popover>

      <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleAvatarChange} />
    </>
  );
}
