<script lang="ts">
	// HOI4 Live: one stream per PC, what each game is doing, a feed like a stream's chat,
	// replays of the games played, and the record. Everything comes from /api/*.
	import { onDestroy } from 'svelte';
	import Chat from '$lib/Chat.svelte';
	import GameCard from '$lib/GameCard.svelte';
	import Games from '$lib/Games.svelte';
	import Replay from '$lib/Replay.svelte';
	import Stage from '$lib/Stage.svelte';
	import StatsPanel from '$lib/StatsPanel.svelte';
	import { get } from '$lib/api';
	import { arena } from '$lib/format';
	import type { Played, Status } from '$lib/types';

	let status: Status | null = $state(null);
	let chosen: string | null = $state(null);
	let tab: 'chat' | 'games' | 'stats' = $state('chat');
	let replaying: Played | null = $state(null);

	// The PC shown: the one chosen, else the first playing, streaming, or any.
	const station = $derived.by(() => {
		if (!status) return null;
		const picked = status.stations.find((s) => s.id === chosen);
		return (
			picked ??
			status.stations.find((s) => s.game) ??
			status.stations.find((s) => s.streaming) ??
			status.stations[0] ??
			null
		);
	});

	const badge = $derived.by(() => {
		if (!status) return { text: 'CONNECTING', kind: '' };
		if (station?.game) return { text: 'LIVE', kind: 'live' };
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

<header>
	<h1>HOI4 Live</h1>
	<span class="badge {badge.kind}">{badge.text}</span>
</header>

{#if status && status.stations.length > 1}
	<nav class="pills">
		{#each status.stations as s (s.id)}
			<button class:on={s.id === station?.id} onclick={() => (chosen = s.id)}>
				<span class="dot" class:live={s.game} class:menus={!s.game && s.streaming}></span>{s.label}{#if s.game}
					· {arena(s.game.arena)}{/if}
			</button>
		{/each}
	</nav>
{/if}

<div class="layout">
	<section class="watch">
		{#if status}
			{#key station?.id}
				<Stage {station} {status} />
			{/key}
			<GameCard {station} />
		{/if}
	</section>
	<section class="side">
		<nav class="pills">
			<button class:on={tab === 'chat'} onclick={() => (tab = 'chat')}>Chat</button>
			<button class:on={tab === 'games'} onclick={() => (tab = 'games')}>Games</button>
			<button class:on={tab === 'stats'} onclick={() => (tab = 'stats')}>Stats</button>
		</nav>
		<div class="panel" hidden={tab !== 'chat'}><Chat /></div>
		{#if tab === 'games'}
			<div class="panel"><Games onopen={(game) => (replaying = game)} /></div>
		{:else if tab === 'stats'}
			<div class="panel"><StatsPanel /></div>
		{/if}
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
	.dot {
		display: inline-block;
		width: 7px;
		height: 7px;
		border-radius: 50%;
		margin-right: 6px;
		background: #555;
		vertical-align: 1px;
	}
	.dot.live {
		background: var(--live);
	}
	.dot.menus {
		background: var(--draw);
	}
	.layout {
		display: flex;
		flex-direction: column;
	}
	@media (min-width: 900px) {
		.layout {
			flex-direction: row;
			align-items: flex-start;
		}
		.watch {
			flex: 1 1 auto;
			min-width: 0;
		}
		.side {
			width: 380px;
			flex: 0 0 380px;
			position: sticky;
			top: 0;
		}
	}
	.side {
		padding-bottom: 16px;
	}
	.panel {
		padding: 0 16px;
	}
</style>
