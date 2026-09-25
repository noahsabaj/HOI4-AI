<script lang="ts">
	// The feed, like a stream's chat: the games narrated as they go, watchers' messages,
	// Claude's, and moments flagged. New lines scroll in while the list is at its bottom.
	import { onDestroy, tick } from 'svelte';
	import { get, post, who } from './api';
	import { hhmm } from './format';
	import type { Message } from './types';

	let messages: Message[] = $state([]);
	let text = $state('');
	let list: HTMLOListElement;
	let last = 0;

	const ICON: Record<string, string> = { event: '🎮 ', flag: '🚩 ' };
	function hue(name: string) {
		let h = 0;
		for (const c of name) h = (h * 31 + c.charCodeAt(0)) % 360;
		return h;
	}

	async function pull() {
		const fresh = await get<Message[]>(`api/chat?after=${last}`);
		if (!fresh?.length) return;
		const bottom = !list || list.scrollHeight - list.scrollTop - list.clientHeight < 40;
		for (const m of fresh) last = Math.max(last, m.id);
		messages = [...messages, ...fresh].slice(-400);
		if (bottom) {
			await tick();
			list.scrollTop = list.scrollHeight;
		}
	}

	async function send(event: SubmitEvent) {
		event.preventDefault();
		const said = text.trim();
		if (!said) return;
		text = '';
		await post('api/chat', { who: who(), text: said });
		pull();
	}

	pull();
	const timer = setInterval(pull, 2000);
	onDestroy(() => clearInterval(timer));
</script>

<ol bind:this={list}>
	{#each messages as m (m.id)}
		<li class={m.kind}>
			<time>{hhmm(m.t)}</time><span
				class="who"
				style:color={m.kind === 'user' ? `hsl(${hue(m.who)},70%,70%)` : null}>{m.who}</span
			>{ICON[m.kind] ?? ''}{m.text}
		</li>
	{/each}
</ol>
<form onsubmit={send}>
	<input bind:value={text} maxlength="300" placeholder="Say something" autocomplete="off" />
	<button>Send</button>
</form>

<style>
	ol {
		list-style: none;
		margin: 0;
		padding: 0;
		max-height: 46vh;
		overflow-y: auto;
		font-size: 14px;
	}
	@media (min-width: 900px) {
		ol {
			max-height: calc(100vh - 170px);
		}
	}
	li {
		padding: 3px 0;
		overflow-wrap: anywhere;
	}
	.who {
		font-weight: 650;
		margin-right: 6px;
	}
	.event {
		color: #b8c4d0;
	}
	.event .who {
		color: var(--muted);
	}
	.claude .who {
		color: var(--claude);
	}
	.flag {
		color: var(--flag);
	}
	form {
		display: flex;
		gap: 6px;
		margin-top: 8px;
	}
	input {
		flex: 1;
		min-width: 0;
		background: var(--panel);
		border: 1px solid var(--line);
		border-radius: 8px;
		padding: 8px 10px;
		font-size: 16px;
	}
	button {
		background: #2c3440;
		border: 1px solid #4a5a70;
		border-radius: 8px;
		padding: 0 14px;
	}
</style>
