import Link from "next/link";
import { DocsShell } from "@/app/docs/_components/DocsShell";
import { CodeBlock } from "@/app/docs/_components/CodeBlock";
import { buildDocsMetadata } from "@/lib/docs/metadata";
import { TRAIN_LAYOUT } from "@/lib/docs/sections/train";

export const metadata = buildDocsMetadata("/docs");

const codeChip = "font-mono text-[13px] bg-stone-200/60 px-1 rounded";
const h2 = "text-xl font-semibold mb-3 mt-10 first:mt-0 tracking-tight";
const p = "text-[15px] text-stone-600 leading-relaxed mb-4 max-w-2xl";

export default function DocsGettingStartedPage() {
  return (
    <DocsShell>
      <article>
        <h1
          className="mb-4 text-4xl font-normal leading-tight tracking-tight sm:text-5xl"
          style={{ fontFamily: "var(--font-host-grotesk), system-ui, sans-serif" }}
        >
          Getting started
        </h1>
        <p className={p}>
          Aquin is a <strong>developer environment and framework</strong> for building and checking
          models. The unit of work is a directory. You grow that folder, run{" "}
          <code className={codeChip}>aq</code> against it, and keep enough on disk that you (or
          someone else) can fork, resume, and audit what happened. Disk is the source of truth. The
          UI is a view. The Unix is the product.
        </p>

        <h2
          className={h2}
          style={{ fontFamily: "var(--font-host-grotesk), system-ui, sans-serif" }}
        >
          How the pieces fit
        </h2>
        <div className="mb-6 max-w-2xl overflow-x-auto">
          <table className="w-full border-collapse text-left text-[14px]">
            <thead>
              <tr className="border-b border-stone-300">
                <th className="py-2 pr-4 font-semibold text-stone-800">Name</th>
                <th className="py-2 font-semibold text-stone-800">What it is</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Aquin", "The product: developer environment and framework."],
                ["aq", "The CLI you type (TypeScript). Quiet, cwd-based, like git."],
                ["Kernel", "Python under aq/kernel/. Fit, eval, serve, hash. Talks to the CLI through files."],
                ["Train", "A directory with instructions.md + recipe.yaml. The real unit of work."],
                ["Agent", "Lives inside that Unix: same cwd, same files, same verbs. Not a side chat."],
              ].map(([name, what]) => (
                <tr key={name} className="border-b border-stone-200/80 align-top">
                  <td className="whitespace-nowrap py-2.5 pr-4 font-mono text-[13px] text-stone-800">
                    {name}
                  </td>
                  <td className="py-2.5 leading-relaxed text-stone-600">{what}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={p}>
          Flow is always the same shape: you work in a train folder, the CLI routes verbs, the kernel
          does the numeric work, artifacts land back on disk. Accounts and docs live at{" "}
          <code className={codeChip}>aq.aquin.app</code>. Auth is identity and tokens only — it does
          not own trains.{" "}
          <Link href="/" className="text-stone-800 underline underline-offset-2">
            Sign in
          </Link>
          .
        </p>

        <h2
          className={h2}
          style={{ fontFamily: "var(--font-host-grotesk), system-ui, sans-serif" }}
        >
          The directory system
        </h2>
        <p className={p}>
          A <strong>train</strong> is any folder with both{" "}
          <code className={codeChip}>instructions.md</code> (what this is for) and{" "}
          <code className={codeChip}>recipe.yaml</code> (the full train API for built-ins).{" "}
          <code className={codeChip}>aq init</code> scaffolds the rest. Convention over registration:
          drop a file in the right slot and it exists. No central registry.
        </p>
        <CodeBlock code={TRAIN_LAYOUT} language="text" label="train folder" />
        <p className={p}>
          <code className={codeChip}>data/</code> and <code className={codeChip}>evals/</code> are
          yours. <code className={codeChip}>methods/</code>, <code className={codeChip}>tools/</code>,{" "}
          <code className={codeChip}>skills/</code>, <code className={codeChip}>schedules/</code>, and{" "}
          <code className={codeChip}>stages/</code> extend the train locally.{" "}
          <code className={codeChip}>jobs/</code> and <code className={codeChip}>artifacts/</code> are
          system-owned. Forking copies the science and skips runtime state so a variant starts clean.
          Full layout:{" "}
          <Link href="/docs/train" className="text-stone-800 underline underline-offset-2">
            Train folder
          </Link>
          .
        </p>

        <h2
          className={h2}
          style={{ fontFamily: "var(--font-host-grotesk), system-ui, sans-serif" }}
        >
          CLI
        </h2>
        <p className={p}>
          <code className={codeChip}>aq</code> is the face. Cwd is the workspace. Help is short. Verbs
          do one thing and compose: <code className={codeChip}>aq train</code> fits,{" "}
          <code className={codeChip}>aq eval</code> scores, <code className={codeChip}>aq serve</code>{" "}
          generates, <code className={codeChip}>aq status</code> /{" "}
          <code className={codeChip}>aq diff</code> read history,{" "}
          <code className={codeChip}>aq job run</code> files long work with resource asks,{" "}
          <code className={codeChip}>aq fork</code> branches a train. Schedules and stages compose
          those verbs further.
        </p>
        <p className={p}>
          On a TTY, bare <code className={codeChip}>aq</code> opens the agent. Without a TTY it prints
          help. Account tokens live under <code className={codeChip}>~/.aquin</code>; provider keys for
          the agent under <code className={codeChip}>~/.aq</code>. Reference:{" "}
          <Link href="/docs/cli" className="text-stone-800 underline underline-offset-2">
            CLI
          </Link>
          .
        </p>

        <h2
          className={h2}
          style={{ fontFamily: "var(--font-host-grotesk), system-ui, sans-serif" }}
        >
          Kernel (Python)
        </h2>
        <p className={p}>
          The kernel is the programmatic engine. TypeScript and Python do not share memory. They talk
          through files:
        </p>
        <ol className="mb-4 max-w-2xl list-decimal space-y-1.5 pl-5 text-[15px] leading-relaxed text-stone-600">
          <li>
            CLI writes <code className={codeChip}>artifacts/request.json</code>
          </li>
          <li>
            Spawns <code className={codeChip}>kernel/run.py</code> on the train path
          </li>
          <li>
            Kernel runs the op (<code className={codeChip}>hash</code>,{" "}
            <code className={codeChip}>train</code>, <code className={codeChip}>eval</code>,{" "}
            <code className={codeChip}>checkpoint</code>, <code className={codeChip}>serve</code>)
          </li>
          <li>
            Writes <code className={codeChip}>artifacts/result.json</code> and appends{" "}
            <code className={codeChip}>metrics.jsonl</code>; live steps stream on stderr
          </li>
        </ol>
        <p className={p}>
          For built-ins you do not write Python. <code className={codeChip}>recipe.yaml</code> is the
          API: tabular (sklearn), transformers, LLM / LoRA / QLoRA. Custom fits go in{" "}
          <code className={codeChip}>methods/&lt;name&gt;.py</code> with{" "}
          <code className={codeChip}>fit(src, rec)</code>. After fit,{" "}
          <code className={codeChip}>artifacts/inspect.md</code> is a capability for seeing the model,
          not a separate product. See{" "}
          <Link href="/docs/recipe" className="text-stone-800 underline underline-offset-2">
            Recipe
          </Link>{" "}
          and{" "}
          <Link href="/docs/methods/tabular" className="text-stone-800 underline underline-offset-2">
            Methods
          </Link>
          .
        </p>

        <h2
          className={h2}
          style={{ fontFamily: "var(--font-host-grotesk), system-ui, sans-serif" }}
        >
          Agent
        </h2>
        <p className={p}>
          The agent is not a chat toy with a private filesystem. It is a resident of the train: same
          cwd, same <code className={codeChip}>recipe.yaml</code>, same tools. It can call{" "}
          <code className={codeChip}>aq</code> verbs, run <code className={codeChip}>tools/</code>, load{" "}
          <code className={codeChip}>skills/</code> (and MCP), and read{" "}
          <code className={codeChip}>memory/</code>. Humans own the eval gate. The agent must not invent
          scores.
        </p>
        <p className={p}>
          Surfaces: interactive <code className={codeChip}>aq</code> /{" "}
          <code className={codeChip}>aq agent</code>, one-shot <code className={codeChip}>aq ask</code>,
          resume via <code className={codeChip}>aq chat</code>, background workers via{" "}
          <code className={codeChip}>aq spawn</code>. Details:{" "}
          <Link href="/docs/agent" className="text-stone-800 underline underline-offset-2">
            Agent
          </Link>{" "}
          and{" "}
          <Link href="/docs/skills" className="text-stone-800 underline underline-offset-2">
            Skills, tools &amp; MCP
          </Link>
          .
        </p>

        <h2
          className={h2}
          style={{ fontFamily: "var(--font-host-grotesk), system-ui, sans-serif" }}
        >
          What “working” means
        </h2>
        <p className={p}>
          A stranger should be able to open a train, point the recipe at their data, set their gate,
          run train then eval, fail, inspect, edit one file, fork, and run again. Long work files as{" "}
          <code className={codeChip}>aq job run</code>. If that path is awkward, the framework is
          unfinished. There is no preset eval zoo. You write the probes.
        </p>

        <h2
          className={h2}
          style={{ fontFamily: "var(--font-host-grotesk), system-ui, sans-serif" }}
        >
          Where to go next
        </h2>
        <ul className="mb-8 max-w-2xl list-none space-y-2 pl-0 text-[15px] leading-relaxed text-stone-600">
          {[
            ["/docs/install", "Install", "curl release or checkout link, doctor, login, providers"],
            ["/docs/train", "Train folder", "slots, artifacts, fork, checkout"],
            ["/docs/recipe", "Recipe", "tabular, transformers, LLM, guard"],
            ["/docs/cli", "CLI", "every verb"],
            ["/docs/jobs", "Jobs", "detached work with resource asks"],
            ["/llms.txt", "llms.txt", "agent scrape index (markdown links)"],
          ].map(([href, label, blurb]) => (
            <li key={href}>
              <Link
                href={href}
                className="font-medium text-stone-900 underline underline-offset-2"
              >
                {label}
              </Link>
              <span className="text-stone-400"> · </span>
              {blurb}
            </li>
          ))}
        </ul>
      </article>
    </DocsShell>
  );
}
