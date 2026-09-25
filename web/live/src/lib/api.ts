// The live view's API, on the same origin as the page.

export async function get<T>(path: string): Promise<T | null> {
	try {
		const answer = await fetch(path, { cache: 'no-store' });
		return answer.ok ? ((await answer.json()) as T) : null;
	} catch {
		return null; // Offline for a moment: the next poll tries again.
	}
}

export async function post<T>(path: string, body: unknown): Promise<T | null> {
	try {
		const answer = await fetch(path, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body)
		});
		return answer.ok ? ((await answer.json()) as T) : null;
	} catch {
		return null;
	}
}

/** The watcher's name in the chat, asked once and kept on this device. */
export function who(): string {
	let name = '';
	try {
		name = localStorage.getItem('who') ?? '';
	} catch {
		/* Private browsing: asked each time. */
	}
	if (!name) {
		name = (prompt('Your name in the chat?') ?? '').trim().slice(0, 24) || 'anon';
		try {
			localStorage.setItem('who', name);
		} catch {
			/* Kept for this visit only. */
		}
	}
	return name;
}

/** Mark a moment for Claude to look at, with a note. */
export async function flag(station: string | null, game: string, seconds: number, replay: boolean) {
	const note = prompt('What should Claude look at here?', '');
	if (note === null) return;
	await post('api/flag', { who: who(), station, game, seconds: Math.round(seconds), note, replay });
}
