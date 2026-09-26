<script lang="ts">
	// HOI4 Live, laid out like a streaming site: the game on the player, what it is doing,
	// the games live now (which PC each is on), the games played to watch again, and the
	// chat beside it all. Everything comes from /api/*.
	import { onDestroy } from 'svelte';
	import Chat from '$lib/Chat.svelte';
	import GameCard from '$lib/GameCard.svelte';
	import LiveNow from '$lib/LiveNow.svelte';
	import Recent from '$lib/Recent.svelte';
	import Replay from '$lib/Replay.svelte';
	import Stage from '$lib/Stage.svelte';
	import StatsPanel from '$lib/StatsPanel.svelte';
	import { get } from '$lib/api';
	import type { Played, Status } from '$lib/types';

	let status = $state<Status | null>(null);
	let chosen: string | null = $state(null);
	let tab: 'chat' | 'stats' = $state('chat');
	let replaying: Played | null = $state(null);

	const live = $derived(status?.stations.filter((s) => s.game) ?? []);

	// The PC on the player: the one picked while it plays, else the first playing,
	// streaming, or any.
	const station = $derived.by(() => {
		if (!status) return null;
		const picked = status.stations.find((s) => s.id === chosen);
		return (
			(picked?.game ? picked : null) ??
			live[0] ??
			picked ??
			status.stations.find((s) => s.streaming) ??
			status.stations[0] ??
			null
		);
	});

	const badge = $derived.by(() => {
		if (!status) return { text: 'CONNECTING', kind: '' };
		if (live.length) return { text: live.length > 1 ? `${live.length} LIVE` : 'LIVE', kind: 'live' };
		if (status.idle_since) return { text: 'STOPPED', kind: 'stopped' };
		return { text: 'BETWEEN GAMES', kind: '' };
	});

	async function poll() {
		const fresh = await get<Status>('api/status');
		if (fresh) status = fresh;
	}
	poll();
	const timer = setInterval(poll, 5000);
	onDestroy(() => clearInterval(timer));
</script>

<svelte:head><title>HOI4 Live</title></svelte:head>
<svelte:document onvisibilitychange={() => !document.hidden && poll()} />

<div class="app">
	<section class="watch">
		<header>
			<h1>HOI4 Live</h1>
			<span class="badge {badge.kind}">{badge.text}</span>
		</header>
		{#if status}
			{#key station?.id}
				<Stage {station} {status} />
			{/key}
			<GameCard {station} />
		{/if}
	</section>

	<aside class="side">
		<nav class="pills">
			<button class:on={tab === 'chat'} onclick={() => (tab = 'chat')}>Chat</button>
			<button class:on={tab === 'stats'} onclick={() => (tab = 'stats')}>Stats</button>
		</nav>
		<div class="panel" hidden={tab !== 'chat'}><Chat /></div>
		{#if tab === 'stats'}
			<div class="panel scroll"><StatsPanel /></div>
		{/if}
	</aside>

	<section class="more">
		{#if live.length}
			<h2>Live now <span class="count">{live.length}</span></h2>
			<LiveNow stations={live} watching={station?.game ? station.id : null} onpick={(id) => {
				chosen = id;
				window.scrollTo({ top: 0, behavior: 'smooth' });
			}} />
		{/if}
		<h2>Recent games</h2>
		<Recent onopen={(game) => (replaying = game)} />
	</section>
</div>

{#if replaying}
	<Replay game={replaying} onclose={() => (replaying = null)} />
{/if}

<style>
	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 10px 16px 6px;
		user-select: none;
	}
	h1 {
		margin: 0;
		font-size: 18px;
		font-weight: 650;
	}
	.badge {
		font-size: 12px;
		font-weight: 700;
		letter-spacing: 0.6px;
		padding: 3px 10px;
		border-radius: 999px;
		background: var(--panel);
		color: var(--muted);
	}
	.badge.live {
		background: var(--live);
		color: #fff;
	}
	.badge.live::before {
		content: '';
		display: inline-block;
		width: 7px;
		height: 7px;
		margin-right: 6px;
		border-radius: 50%;
		background: #fff;
		vertical-align: 1px;
		animation: pulse 1.6s ease-in-out infinite;
	}
	.badge.stopped {
		background: #3a2a1a;
		color: var(--flag);
	}
	@keyframes pulse {
		50% {
			opacity: 0.25;
		}
	}
	.app {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		grid-template-areas: 'watch' 'side' 'more';
	}
	.watch {
		grid-area: watch;
		min-width: 0;
	}
	.more {
		grid-area: more;
		min-width: 0;
		padding-bottom: 32px;
	}
	/* The chat keeps its box in view: the list scrolls, never the box away. */
	.side {
		grid-area: side;
		display: flex;
		flex-direction: column;
		height: 62dvh;
		min-height: 0;
		border-top: 1px solid var(--line);
		border-bottom: 1px solid var(--line);
		padding-top: 8px;
	}
	@media (min-width: 900px) {
		.app {
			grid-template-columns: minmax(0, 1fr) 380px;
			grid-template-rows: auto 1fr;
			grid-template-areas: 'watch side' 'more side';
		}
		.side {
			align-self: start;
			position: sticky;
			top: env(safe-area-inset-top);
			height: calc(100dvh - env(safe-area-inset-top) - env(safe-area-inset-bottom));
			border: 0;
			border-left: 1px solid var(--line);
		}
	}
	.panel {
		flex: 1;
		min-height: 0;
		display: flex;
		flex-direction: column;
		padding: 0 16px 12px;
	}
	.panel.scroll {
		overflow-y: auto;
		display: block;
	}
	h2 {
		font-size: 17px;
		font-weight: 650;
		margin: 22px 16px 10px;
	}
	.count {
		font-size: 12px;
		font-weight: 700;
		background: var(--live);
		color: #fff;
		border-radius: 999px;
		padding: 1px 8px;
		vertical-align: 2px;
		margin-left: 4px;
	}
</style>
