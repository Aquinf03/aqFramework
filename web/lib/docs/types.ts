export type ToolFlag = {
  name: string;
  required?: boolean;
  description: string;
};

export type ToolDoc = {
  command: string;
  agentTool?: string;
  description: string;
  flags: ToolFlag[];
  /** One command per string. Do not stack shell commands in a single example. */
  example?: string | string[];
  notes?: string;
};

export type DocSection = {
  title: string;
  intro: string;
  prerequisite?: string;
  tools: ToolDoc[];
};

export type NavIconKey =
  | "rocket"
  | "terminal"
  | "magnifyingGlass"
  | "cube"
  | "chartLine"
  | "clipboard"
  | "brain"
  | "circlesThree";

export type NavItem = {
  label: string;
  href?: string;
  icon?: NavIconKey;
  children?: NavItem[];
};
