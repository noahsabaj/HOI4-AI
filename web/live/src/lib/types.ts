// What the live view's server answers (src/hoi4_arena/live/app.py).

export type Side = 'BLU' | 'RED';
export type Result = 'win' | 'loss' | 'timeout';

export interface Order {
	frame?: number;
	seconds: number;
	order: string;
	attack?: string;
}

export interface SideReport {
	surrender?: number;
	states?: number;
	owned?: number;
	divisions?: number;
	strength?: number;
	casualties?: number;
}

export interface Plan {
	variant?: string;
	wait?: number;
	redraw?: number;
	guard?: number;
	depth?: number;
	conscription?: string;
	recruit?: number;
	[key: string]: unknown;
}

/** A game being played now. */
export interface Card {
	game: string;
	run: string;
	station: string;
	arena: string | null;
	side: Side;
	plan: Plan;
	started_unix: number | null;
	elapsed: number | null;
	date: string | null;
	sides: Partial<Record<Side, SideReport>>;
	declarer: Side | null;
	orders: Order[];
	kicks: number;
	milestones: Record<string, number> | null;
	winner: string | null;
}

/** A game already played. */
export interface Played {
	run: string;
	game: string;
	station: string;
	arena: string | null;
	side: Side;
	plan: string | null;
	result: Result;
	seconds: number;
	started_unix: number | null;
	ended_unix: number | null;
	milestones: Record<string, number> | null;
}

export interface Station {
	id: string;
	label: string;
	streaming: boolean;
	fps: number;
	game: Card | null;
	last: Played | null;
}

export interface Status {
	updated: number;
	stations: Station[];
	running: { kind: string; output: string }[];
	last_end: number | null;
	idle_since: number | null;
	record: Played[];
}

export interface Message {
	id: number;
	t: number;
	who: string;
	kind: 'event' | 'user' | 'claude' | 'flag';
	text: string;
	station?: string;
	game?: string;
}

export type Tally = Record<Result, number>;

export interface Training {
	run: string;
	epoch: number;
	step: number;
	loss: number;
	updated_unix: number;
}

export interface Stats {
	hours: number;
	by_arena: Record<string, Record<string, Tally>>;
	training: Training[];
}

export type ReplayAnswer =
	| { state: 'ready'; url: string; orders?: Order[] }
	| { state: 'working'; progress: number }
	| { state: 'busy' }
	| { state: 'error'; error: string };
