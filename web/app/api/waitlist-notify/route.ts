import { NextRequest, NextResponse } from "next/server";
import nodemailer from "nodemailer";

export async function POST(req: NextRequest) {
  const { name, email, useCase, role, company, twitter, linkedin } = await req.json();

  const transporter = nodemailer.createTransport({
    host: process.env.SMTP_HOST,
    port: Number(process.env.SMTP_PORT ?? 587),
    secure: process.env.SMTP_SECURE === "true",
    auth: {
      user: process.env.SMTP_USER,
      pass: process.env.SMTP_PASS,
    },
  });

  await transporter.sendMail({
    from: process.env.SMTP_FROM,
    to: process.env.SMTP_USER,
    subject: `New waitlist signup: ${name || email}`,
    html: `
      <div style="font-family: sans-serif; max-width: 600px;">
        <h2 style="color: #1c1917;">New waitlist request</h2>
        <table style="width:100%; border-collapse: collapse; margin-top: 16px;">
          <tr>
            <td style="padding: 8px 12px; background: #f5f5f4; font-weight: 600; width: 140px;">Name</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e7e5e4;">${name || "—"}</td>
          </tr>
          <tr>
            <td style="padding: 8px 12px; background: #f5f5f4; font-weight: 600;">Email</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e7e5e4;">${email}</td>
          </tr>
          <tr>
            <td style="padding: 8px 12px; background: #f5f5f4; font-weight: 600;">Role</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e7e5e4;">${role || "—"}</td>
          </tr>
          <tr>
            <td style="padding: 8px 12px; background: #f5f5f4; font-weight: 600;">Company</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e7e5e4;">${company || "—"}</td>
          </tr>
          <tr>
            <td style="padding: 8px 12px; background: #f5f5f4; font-weight: 600; vertical-align: top;">Use case</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e7e5e4;">${useCase || "—"}</td>
          </tr>
          <tr>
            <td style="padding: 8px 12px; background: #f5f5f4; font-weight: 600;">X / Twitter</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e7e5e4;">${twitter ? `<a href="${twitter}">${twitter}</a>` : "—"}</td>
          </tr>
          <tr>
            <td style="padding: 8px 12px; background: #f5f5f4; font-weight: 600;">LinkedIn</td>
            <td style="padding: 8px 12px;">${linkedin ? `<a href="${linkedin}">${linkedin}</a>` : "—"}</td>
          </tr>
        </table>
        <p style="margin-top: 24px; color: #78716c; font-size: 13px;">
          To approve, set <code>is_approved = true</code> in the profiles table for this user's email.
        </p>
      </div>
    `,
  });

  return NextResponse.json({ ok: true });
}
