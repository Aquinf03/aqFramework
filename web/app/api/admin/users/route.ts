import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";
import { supabaseService } from "@/lib/supabase/service";

const ADMIN_EMAILS = [
  "aquin@aquin.app",
  "ash@aquin.app",
  "sambhav@aquin.app",
  "paul@aquin.app",
  "sachin@aquin.app",
  "neha@aquin.app"
];

async function isAdmin(): Promise<boolean> {
  try {
    const supabase = await createClient();
    const { data: { user } } = await supabase.auth.getUser();
    return !!user?.email && ADMIN_EMAILS.includes(user.email);
  } catch {
    return false;
  }
}

export async function GET() {
  if (!(await isAdmin())) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const { data, error } = await supabaseService
    .from("profiles")
    .select("id, email, name, avatar_url, is_approved, waitlist_role, waitlist_use_case, waitlist_company, waitlist_twitter, waitlist_linkedin, created_at")
    .order("created_at", { ascending: false });

  if (error) {
    console.error("[admin/users] query error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ users: data });
}

export async function PATCH(req: NextRequest) {
  if (!(await isAdmin())) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const { id, is_approved } = await req.json();
  const { error } = await supabaseService
    .from("profiles")
    .update({ is_approved })
    .eq("id", id);

  if (error) {
    console.error("[admin/users] update error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ ok: true });
}
