/** Proxy /releases/* on aq.aquin.app → R2 bucket aqfw-releases. */

const PREFIX = "/releases/";

export interface Env {
  RELEASES: R2Bucket;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (!url.pathname.startsWith(PREFIX)) {
      return new Response("Not found", { status: 404 });
    }

    const key = url.pathname.slice(PREFIX.length);
    if (!key || key.includes("..")) {
      return new Response("Bad path", { status: 400 });
    }

    const object = await env.RELEASES.get(key);
    if (!object) {
      return new Response("Not found", { status: 404 });
    }

    const headers = new Headers();
    object.writeHttpMetadata(headers);
    headers.set("Cache-Control", "public, max-age=300");
    headers.set("Access-Control-Allow-Origin", "*");

    return new Response(object.body, { headers });
  },
};
