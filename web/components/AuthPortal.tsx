/* eslint-disable @next/next/no-img-element */
"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { Check, CircleNotch, Copy, Eye, EyeSlash } from "@phosphor-icons/react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import CliTokenSection from "@/components/account/CliTokenSection";
import ProfileChip from "@/components/account/ProfileChip";
import { AquinBrand } from "@/components/ui/AquinBrand";
import { PoliciesDropdown } from "@/components/PoliciesDropdown";
import {
  ghostBtnCls,
  inputCls,
  inputShellCls,
  inputInnerCls,
  continueInInputBtnCls,
  labelCls,
  messageErrorCls,
  messageSuccessCls,
  primaryBtnCls,
  secondaryBtnCls,
} from "@/lib/auth-ui";

type AuthStep = "email" | "password" | "signup" | "signup-password" | "ready" | "desktop";
type DesktopPhase = "minting" | "ready" | "error";

function codeFromDeepLink(link: string): string | null {
  try {
    return new URL(link).searchParams.get("code");
  } catch {
    return null;
  }
}

function AuthPortalInner() {
  const { user, loading: authLoading } = useAuth();
  const supabase = createClient();
  const router = useRouter();
  const searchParams = useSearchParams();
  const viewDesktop = searchParams.get("view") === "desktop";
  const client = (searchParams.get("client") || "").toLowerCase();
  const isCli = client === "cli" || client === "";

  const [step, setStep] = useState<AuthStep>("email");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: "error" | "success"; text: string } | null>(null);

  const [desktopPhase, setDesktopPhase] = useState<DesktopPhase>("minting");
  const [desktopError, setDesktopError] = useState<string | null>(null);
  const [deepLink, setDeepLink] = useState<string | null>(null);
  const [code, setCode] = useState<string | null>(null);
  const [copied, setCopied] = useState<"code" | "link" | null>(null);
  const minted = useRef(false);

  const desktopQuery = useMemo(() => {
    const q = new URLSearchParams();
    q.set("view", "desktop");
    if (client) q.set("client", client);
    return `/?${q.toString()}`;
  }, [client]);

  useEffect(() => {
    if (authLoading) return;
    if (!user) {
      if (viewDesktop) setStep("email");
      return;
    }
    if (user.email) setEmail(user.email);
    if (viewDesktop) {
      setStep("desktop");
      return;
    }
    setStep("ready");
  }, [user, authLoading, viewDesktop]);

  const mintDesktopCode = useCallback(async () => {
    setDesktopPhase("minting");
    setDesktopError(null);
    setCopied(null);
    try {
      const res = await fetch("/api/auth/desktop/mint", { method: "POST" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(
          typeof data.error === "string" ? data.error : "Could not prepare sign-in code.",
        );
      }
      const link = String(data.deep_link || "");
      const raw =
        typeof data.code === "string" && data.code ? data.code : codeFromDeepLink(link);
      if (!raw) throw new Error("Invalid desktop handoff.");
      setCode(raw);
      setDeepLink(link.startsWith("aquin://") ? link : `aquin://auth?code=${encodeURIComponent(raw)}`);
      setDesktopPhase("ready");
    } catch (e) {
      setDesktopError(e instanceof Error ? e.message : "Something went wrong.");
      setDesktopPhase("error");
    }
  }, []);

  useEffect(() => {
    if (step !== "desktop" || !user || minted.current) return;
    minted.current = true;
    void mintDesktopCode();
  }, [step, user, mintDesktopCode]);

  const checkEmailExists = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);
    try {
      const { data, error } = await supabase
        .from("profiles")
        .select("email")
        .eq("email", email)
        .maybeSingle();
      if (error) throw error;
      setStep(data ? "password" : "signup");
    } catch (err) {
      setMessage({ type: "error", text: err instanceof Error ? err.message : "An error occurred" });
    } finally {
      setLoading(false);
    }
  };

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);
    try {
      const { data, error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) throw error;
      const { data: profile } = await supabase
        .from("profiles")
        .select("id")
        .eq("id", data.user.id)
        .maybeSingle();
      if (!profile) {
        await supabase.from("profiles").insert({
          id: data.user.id,
          email: data.user.email!,
        });
      }
      if (viewDesktop) {
        router.replace(desktopQuery);
        return;
      }
      setStep("ready");
    } catch (err) {
      setMessage({ type: "error", text: err instanceof Error ? err.message : "An error occurred" });
    } finally {
      setLoading(false);
    }
  };

  const handleSignupNext = (e: React.FormEvent) => {
    e.preventDefault();
    setMessage(null);
    setStep("signup-password");
  };

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: { data: { name: name || null } },
      });
      if (error) throw error;
      if (data.user) {
        const { error: profileError } = await supabase.from("profiles").insert({
          id: data.user.id,
          email: data.user.email!,
          name: name || null,
          avatar_url: null,
        });
        if (profileError) throw profileError;

        const { error: signInError } = await supabase.auth.signInWithPassword({ email, password });
        if (signInError) throw signInError;
        if (viewDesktop) {
          router.replace(desktopQuery);
          return;
        }
        setStep("ready");
      }
    } catch (err) {
      setMessage({ type: "error", text: err instanceof Error ? err.message : "An error occurred" });
    } finally {
      setLoading(false);
    }
  };

  const handleForgotPassword = async () => {
    if (!email) {
      setMessage({ type: "error", text: "Please enter your email first" });
      return;
    }
    setLoading(true);
    setMessage(null);
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/auth/reset-password`,
      });
      if (error) throw error;
      setMessage({ type: "success", text: "Password reset link sent to your email." });
    } catch (err) {
      setMessage({ type: "error", text: err instanceof Error ? err.message : "An error occurred" });
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setStep("email");
    setPassword("");
    setName("");
    setMessage(null);
  };

  const copyText = async (which: "code" | "link", text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(which);
      window.setTimeout(() => setCopied(null), 1600);
    } catch {
      /* select-all still works */
    }
  };

  const openDesktop = () => {
    if (!deepLink) return;
    const a = document.createElement("a");
    a.href = deepLink;
    a.rel = "noopener noreferrer";
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#f5f5f3]">
        <CircleNotch className="animate-spin h-6 w-6 text-stone-400" weight="bold" />
      </div>
    );
  }

  return (
    <div className="relative min-h-screen bg-[#f5f5f3]">
      <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-start gap-4 px-6 py-4 bg-[#f5f5f3]/95 backdrop-blur-sm">
        <AquinBrand size="sm" href="/" />
        <PoliciesDropdown />
      </div>

      <div className="flex min-h-screen items-center justify-center overflow-y-auto px-4 py-24">
        <div className="w-full max-w-md">
          {step === "ready" && (
            <div className="space-y-6">
              <div className="flex flex-col items-stretch text-center gap-5">
                <div>
                  <h2 className="font-host-grotesk text-2xl font-semibold tracking-[-0.03em] text-stone-900">You&apos;re signed in</h2>
                  <p className="text-sm text-stone-500 mt-2">
                    Use the desktop app for AI and workspaces. CLI still uses a token below.
                  </p>
                </div>
                <ProfileChip fullWidth />
              </div>

              <Link
                href="/?view=desktop"
                className={primaryBtnCls + " transition-colors"}
              >
                Open Aquin Desktop
              </Link>

              <div className="text-left">
                <CliTokenSection />
              </div>
            </div>
          )}

          {step === "desktop" && user && (
            <div className="space-y-6 text-center">
              <div>
                <h1 className="font-host-grotesk text-2xl font-semibold tracking-[-0.03em] text-stone-900">
                  {isCli ? "Sign in for Aquin CLI" : "Sign in for CLI / desktop"}
                </h1>
                <p className="text-sm text-stone-500 mt-2">
                  {isCli
                    ? "Copy the code and paste it into the terminal where aq login is waiting."
                    : "Copy the code for the CLI, or open the desktop app with the button below."}
                </p>
              </div>

              <ProfileChip fullWidth />

              {desktopPhase === "minting" && (
                <div className="flex justify-center py-2">
                  <CircleNotch className="animate-spin h-5 w-5 text-stone-400" weight="bold" />
                </div>
              )}

              {desktopError && (
                <p className={messageErrorCls + " text-left"}>
                  {desktopError}
                </p>
              )}

              {desktopPhase === "ready" && code && (
                <div className="space-y-3 text-left">
                  <p className="text-sm text-stone-500 text-center">
                    In the terminal where <span className="font-mono text-stone-700">aq login</span> is waiting,
                    paste the code or the full <span className="font-mono">aquin://</span> link.
                  </p>
                  <div className="rounded-xl border border-stone-200 bg-white/80 px-3 py-2.5 space-y-2">
                    <p className="text-[10px] font-medium uppercase tracking-wide text-stone-400">Code</p>
                    <p className="text-[11px] font-mono text-stone-800 break-all select-all leading-relaxed">
                      {code}
                    </p>
                    <button
                      type="button"
                      className="w-full inline-flex items-center justify-center gap-1.5 rounded-lg bg-black px-3 py-1.5 text-xs font-medium text-white hover:bg-black/90"
                      onClick={() => void copyText("code", code)}
                    >
                      {copied === "code" ? <Check className="h-3.5 w-3.5" weight="bold" /> : <Copy className="h-3.5 w-3.5" weight="bold" />}
                      {copied === "code" ? "Copied" : "Copy code"}
                    </button>
                  </div>
                  {deepLink && (
                    <div className="rounded-xl border border-stone-200 bg-white/80 px-3 py-2.5 space-y-2">
                      <p className="text-[10px] font-medium uppercase tracking-wide text-stone-400">aquin:// link</p>
                      <p className="text-[11px] font-mono text-stone-700 break-all select-all">{deepLink}</p>
                      <button
                        type="button"
                        className={secondaryBtnCls}
                        onClick={() => void copyText("link", deepLink)}
                      >
                        {copied === "link" ? <Check className="h-3.5 w-3.5" weight="bold" /> : <Copy className="h-3.5 w-3.5" weight="bold" />}
                        {copied === "link" ? "Copied" : "Copy link"}
                      </button>
                    </div>
                  )}
                </div>
              )}

              {desktopPhase === "error" && (
                <button
                  type="button"
                  onClick={() => {
                    minted.current = false;
                    void mintDesktopCode();
                  }}
                  className={primaryBtnCls}
                >
                  Try again
                </button>
              )}

              {desktopPhase === "ready" && deepLink && !isCli && (
                <button
                  type="button"
                  onClick={openDesktop}
                  className={primaryBtnCls}
                >
                  Open Aquin Desktop
                </button>
              )}

              {desktopPhase === "ready" && deepLink && isCli && (
                <button
                  type="button"
                  onClick={openDesktop}
                  className={ghostBtnCls}
                >
                  Open Aquin Desktop instead
                </button>
              )}

              <Link href="/" className="block text-xs text-stone-400 hover:text-stone-600">
                &larr; Back to account
              </Link>
            </div>
          )}

          {step === "email" && (
            <div className="mx-auto flex w-full max-w-md flex-col items-center space-y-7">
              <h1 className="font-host-grotesk text-center text-3xl font-semibold tracking-[-0.03em] text-stone-900 leading-tight">
                {viewDesktop ? "Sign in for Aquin CLI" : "Get Started with aq"}
              </h1>
              {viewDesktop && (
                <p className="text-center text-sm text-stone-500">
                  Use your aquin.app account, then paste the code into aq login.
                </p>
              )}

              <form className="w-full space-y-3.5" onSubmit={checkEmailExists}>
                <div className={inputShellCls}>
                  <input
                    id="email"
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className={inputInnerCls}
                    placeholder="you@example.com"
                    aria-label="Email"
                    autoFocus
                  />
                  <button
                    type="submit"
                    disabled={loading}
                    className={continueInInputBtnCls}
                  >
                    {loading ? (
                      <CircleNotch className="size-4 animate-spin" weight="bold" />
                    ) : (
                      "Continue"
                    )}
                  </button>
                </div>

                {message && (
                  <p className={messageErrorCls}>
                    {message.text}
                  </p>
                )}
              </form>
            </div>
          )}

          {step === "password" && (
            <div className="space-y-7">
              <div>
                <h1 className="font-host-grotesk text-3xl font-semibold tracking-[-0.03em] text-stone-900">Enter password</h1>
              </div>

              <form className="space-y-3.5" onSubmit={handleSignIn}>
                <div>
                  <div className="flex items-center gap-2">
                    <input
                      type="email"
                      value={email}
                      disabled
                      aria-label="Email"
                      className="flex-1 px-4 py-3 rounded-xl border border-stone-200 bg-stone-100 text-stone-500 text-sm"
                    />
                    <button type="button" onClick={resetForm} className="text-xs font-host-grotesk text-stone-500 hover:text-stone-900 whitespace-nowrap transition-colors">
                      change
                    </button>
                  </div>
                </div>

                <div>
                  <label htmlFor="password" className={labelCls}>Password</label>
                  <div className="relative">
                    <input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      required
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className={inputCls + " pr-12"}
                      placeholder="••••••••"
                      autoFocus
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute inset-y-0 right-0 pr-4 flex items-center text-stone-400 hover:text-stone-700 transition-colors"
                    >
                      {showPassword ? <EyeSlash className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>

                <div className="flex justify-end">
                  <button type="button" onClick={handleForgotPassword} className="text-xs font-host-grotesk text-stone-400 hover:text-stone-700 transition-colors">
                    forgot password?
                  </button>
                </div>

                {message && (
                  <p className={message.type === "error" ? messageErrorCls : messageSuccessCls}>
                    {message.text}
                  </p>
                )}

                <button
                  type="submit"
                  disabled={loading}
                  className={primaryBtnCls + " transition-colors"}
                >
                  {loading ? <CircleNotch className="animate-spin h-4 w-4" weight="bold" /> : null}
                  {loading ? "Signing in…" : "Sign In"}
                </button>
              </form>
            </div>
          )}

          {step === "signup" && (
            <div className="space-y-7">
              <div>
                <h1 className="font-host-grotesk text-3xl font-semibold tracking-[-0.03em] text-stone-900 leading-tight">Create account</h1>
                <p className="text-sm text-stone-500 mt-1.5 leading-relaxed">
                  Set up your Aquin account to get started.
                </p>
              </div>

              <form className="space-y-3.5" onSubmit={handleSignupNext}>
                <div>
                  <div className="flex items-center gap-2">
                    <input type="email" value={email} disabled aria-label="Email" className="flex-1 px-4 py-3 rounded-xl border border-stone-200 bg-stone-100 text-stone-500 text-sm" />
                    <button type="button" onClick={resetForm} className="text-xs font-host-grotesk text-stone-500 hover:text-stone-900 whitespace-nowrap transition-colors">
                      change
                    </button>
                  </div>
                </div>

                <div>
                  <label htmlFor="name" className={labelCls}>Name</label>
                  <input id="name" type="text" value={name} onChange={(e) => setName(e.target.value)} className={inputCls} placeholder="Jane Smith" autoFocus />
                </div>

                <button type="submit" className={primaryBtnCls + " transition-colors"}>
                  Continue
                </button>
              </form>
            </div>
          )}

          {step === "signup-password" && (
            <div className="space-y-7">
              <div>
                <p className="text-xs font-host-grotesk uppercase tracking-widest text-stone-400 mb-3">Almost there</p>
                <h1 className="font-host-grotesk text-3xl font-semibold tracking-[-0.03em] text-stone-900 leading-tight">Choose a password</h1>
                <p className="text-sm text-stone-500 mt-1.5">Secure your new Aquin account.</p>
              </div>

              <form className="space-y-3.5" onSubmit={handleSignUp}>
                <div>
                  <label htmlFor="signup-password" className={labelCls}>Password</label>
                  <div className="relative">
                    <input
                      id="signup-password"
                      type={showPassword ? "text" : "password"}
                      required
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className={inputCls + " pr-12"}
                      placeholder="••••••••"
                      minLength={6}
                      autoFocus
                    />
                    <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute inset-y-0 right-0 pr-4 flex items-center text-stone-400 hover:text-stone-700 transition-colors">
                      {showPassword ? <EyeSlash className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                  <p className="text-xs font-host-grotesk text-stone-400 mt-1.5">min. 6 characters</p>
                </div>

                {message && (
                  <p className={message.type === "error" ? messageErrorCls : messageSuccessCls}>
                    {message.text}
                  </p>
                )}

                <button
                  type="submit"
                  disabled={loading}
                  className={primaryBtnCls + " transition-colors"}
                >
                  {loading ? <CircleNotch className="animate-spin h-4 w-4" weight="bold" /> : null}
                  {loading ? "Creating account…" : "Create account"}
                </button>

                <button type="button" onClick={() => setStep("signup")} className="w-full py-2 text-xs font-host-grotesk text-stone-400 hover:text-stone-700 transition-colors">
                  &larr; back
                </button>
              </form>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function AuthPortal() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center bg-[#f5f5f3]">
          <CircleNotch className="animate-spin h-6 w-6 text-stone-400" weight="bold" />
        </div>
      }
    >
      <AuthPortalInner />
    </Suspense>
  );
}
