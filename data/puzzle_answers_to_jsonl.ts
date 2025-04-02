#!/usr/bin/env -S deno run --allow-read --allow-write
import { parse } from "https://deno.land/std/flags/mod.ts";

// Parse command line arguments
const args = parse(Deno.args, {
  string: ["input", "output"],
  default: {
    input: "./raw/20250329/gtg_puzzles.ts",
    output: "./20250329/gtg_puzzles.jsonl"
  },
  alias: {
    i: "input",
    o: "output"
  }
});

// Import the answers dynamically based on input path
const importPath = args.input.startsWith("./") 
  ? new URL(args.input, import.meta.url).href 
  : args.input;

const { answers } = await import(importPath);

// Define the output schema based on schema.py
interface Game {
  id: string;
  description: string;
  answers: string[];
  franchise: string;
  submitted_by: string;
  release_year: string;
  metacritic_score: string;
  genre: string;
  console_platform: string;
  developer: string;
}

// Process each answer and convert to the Game schema
const games: Game[] = Object.entries(answers).map(([id, answer]) => {
  return {
    id,
    description: answer.content,
    answers: answer.answers,
    franchise: answer.franchise,
    submitted_by: answer.submitted_by,
    release_year: answer.release_year,
    metacritic_score: answer.metacritic_score,
    genre: answer.genre,
    console_platform: answer.console_platform,
    developer: answer.developer,
  };
});

// Convert to JSONL format (one JSON object per line)
const jsonlContent = games.map(game => JSON.stringify(game)).join("\n");

// Write to output file
const outputPath = args.output;

try {
  // Ensure the output directory exists if needed
  const outputDir = outputPath.substring(0, outputPath.lastIndexOf("/"));
  if (outputDir) {
    await Deno.mkdir(outputDir, { recursive: true }).catch(() => {});
  }
  
  // Write the JSONL file
  await Deno.writeTextFile(outputPath, jsonlContent);
  console.log(`Successfully wrote ${games.length} games to ${outputPath}`);
} catch (error) {
  console.error("Error writing JSONL file:", error);
}
