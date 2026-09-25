import type { Played, Plan } from './types';

export const SIDE: Record<string, string> = { BLU: 'Blue', RED: 'Red' };
export const VERDICT = { win: 'won', loss: 'lost', timeout: 'timed out' } as const;
export const ORDERS: Record<string, string> = {
	army: 'formed the army',
	general: 'gave it a general',
	front: 'drew the front',
	offensive: 'drew the offensive',
	activate: 'launched the attack',
	run: 'started the game',
	law: 'raised conscription',
	clear: 'cleared orders',
	pause: 'paused to redraw',
	guard: 'redrew round an incursion',
	pocket: 'left a pocket behind',
	reinforce: 'reinforced',
	recruit: 'queued divisions'
};
/** A learned policy's setup steps, as the live games count them. */
export const MILESTONES: Record<string, string> = {
	alert: 'alert',
	plus: 'create army',
	portrait: 'general',
	law_slot: 'law',
	front: 'fronts',
	offensive: 'offensives',
	arrow: 'execute'
};

/** "arena-plains-v6" as "plains". */
export const arena = (name: string | null | undefined) =>
	(name ?? '?').replace(/^arena-/, '').replace(/-v\d+$/, '');

export const clock = (seconds: number | null | undefined) =>
	seconds == null
		? ''
		: `${Math.floor(seconds / 60)}:${String(Math.floor(seconds) % 60).padStart(2, '0')}`;

export const hhmm = (t: number | null | undefined) =>
	t ? new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';

export function ago(t: number | null | undefined) {
	if (!t) return '';
	const s = Date.now() / 1000 - t;
	if (s < 90) return 'just now';
	if (s < 5400) return `${Math.round(s / 60)} min ago`;
	if (s < 172800) return `${Math.round(s / 3600)} h ago`;
	return `${Math.round(s / 86400)} days ago`;
}

/** The time, with the weekday when it was not today. */
export function day(t: number | null | undefined) {
	if (!t) return '';
	const d = new Date(t * 1000);
	if (d.toDateString() === new Date().toDateString()) return hhmm(t);
	return `${d.toLocaleDateString([], { weekday: 'short' })} ${hhmm(t)}`;
}

/** "13:00, 1 February, 1937" as "1 February, 1937". */
export const gameDate = (date: string | null | undefined) =>
	(date ?? '').replace(/^\s*\d+:\d+,\s*/, '');

export const lastLine = (g: Played) =>
	`${arena(g.arena)} as ${SIDE[g.side] ?? g.side}, ${VERDICT[g.result]} in ${clock(g.seconds)} (${day(g.started_unix)})`;

export function planTags(plan: Plan | null | undefined): string[] {
	if (!plan) return [];
	const tags: string[] = [];
	if (plan.variant) tags.push(`${plan.variant} ${plan.variant === 'learned' ? 'player' : 'plan'}`);
	if (plan.wait != null) tags.push(`hold ${plan.wait} s`);
	if (plan.redraw != null) tags.push(`redraw ${plan.redraw} s`);
	if (plan.guard != null) tags.push(`guard ${plan.guard}`);
	if (plan.depth != null) tags.push(`depth ${Number(plan.depth).toFixed(2)}`);
	if (plan.conscription) tags.push(plan.conscription.replace('_', ' '));
	if (plan.recruit) tags.push(`recruit ${plan.recruit}`);
	return tags;
}

// Safari, and every browser on an iPhone or iPad, plays HLS itself; elsewhere hls.js does
// (Chromium answers "maybe" for HLS without playing it).
export function nativeHls() {
	const ua = navigator.userAgent;
	return (
		/iPhone|iPad|iPod/.test(ua) ||
		(/Macintosh/.test(ua) &&
			(navigator.maxTouchPoints > 1 || !/Chrome|Chromium|Firefox|Edg/.test(ua)))
	);
}
