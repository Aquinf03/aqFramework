import { existsSync } from "node:fs"
import { cp, mkdir } from "node:fs/promises"
import path from "node:path"
import { jobHome, resolveJobId } from "../job/job.js"
import { assertTrain, isTrain } from "../core/schema.js"

const USAGE = "usage: aq checkout <id>   or  aq checkout <id> <dest>"

export function parseCheckoutArgs(argv: string[]): { dir: string; id: string; dest?: string } {
  if (argv.length === 1) return { dir: ".", id: argv[0] }
  if (argv.length === 2) {
    if (isTrain(path.resolve(argv[0]))) return { dir: argv[0], id: argv[1] }
    return { dir: ".", id: argv[0], dest: argv[1] }
  }
  if (argv.length === 3) return { dir: argv[0], id: argv[1], dest: argv[2] }
  throw new Error(USAGE)
}

export async function checkout(
  dirArg: string,
  idArg: string,
  destArg?: string,
): Promise<{ id: string; dest: string }> {
  const train = assertTrain(dirArg)
  const id = await resolveJobId(train, idArg)
  const tree = path.join(jobHome(train, id), "tree")
  if (!existsSync(tree)) {
    throw new Error(`no tree for job ${id}`)
  }

  if (!destArg) {
    await cp(tree, train, { recursive: true })
    return { id, dest: train }
  }

  const dest = path.resolve(destArg)
  if (existsSync(dest)) throw new Error(`already exists: ${dest}`)
  await mkdir(path.dirname(dest), { recursive: true })
  await cp(tree, dest, { recursive: true })
  await mkdir(path.join(dest, "jobs"), { recursive: true })
  await cp(jobHome(train, id), path.join(dest, "jobs", id), { recursive: true })
  return { id, dest }
}
