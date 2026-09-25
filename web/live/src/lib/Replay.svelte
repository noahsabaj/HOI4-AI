<script lang="ts">
	// A game played, watched again: its replay is made from the recording when asked for
	// (a few seconds), played at 1x to 8x, with its orders to jump to and a flag button.
	import { onDestroy } from 'svelte';
	import { flag, get } from './api';
	import { ORDERS, SIDE, VERDICT, arena, clock, day } from './format';
	import type { Order, Played, ReplayAnswer } from './types';

	let { game, onclose }: { game: Played; onclose: () => void } = $props();

	let video: HTMLVideoElement;
	let url: string | null = $state(null);
	let orders: Order[] = $state([]);
	let note = $state('Preparing the replay…');
	let speed = $state(1);
	let timer: ReturnType<typeof setTimeout> | undefined;
	let closed = false;

	async function ask() {
		if (closed) return;
		const answer = await get<ReplayAnswer>(`api/replay?game=${encodeURIComponent(game.game)}`);
		if (closed) return;
		if (!answer) {
			note = 'No replay: offline';
		} else if (answer.state === 'ready') {
			note = 'Recorded at 5 frames a second.';
			url = answer.url;
			orders = answer.orders ?? [];
		} else if (answer.state === 'working' || answer.state === 'busy') {
			note =
				answer.state === 'busy'
					? 'Another replay is being made…'
					: `Preparing the replay… ${Math.round(100 * answer.progress)}%`;
			timer = setTimeout(ask, 1000);
		} else {
			note = `No replay: ${answer.error}`;
		}
	}
	ask();

	$effect(() => {
		if (video) video.playbackRate = speed;
	});

	function jump(seconds: number) {
		video.currentTime = Math.max(0, seconds - 2);
		video.play().catch(() => {});
	}

	onDestroy(() => {
		closed = true;
		clearTimeout(timer);
	});
</script>

<div class="backdrop">
	<div class="sheet">
		<div class="bar">
			<strong
				>{arena(game.arena)} · {SIDE[game.side] ?? game.side} · {VERDICT[game.result]} in {clock(
					game.seconds
				)} · {day(game.started_unix)}</strong
			>
			<button class="chip" onclick={onclose}>Close</button>
		</div>
		<!-- svelte-ignore a11y_media_has_caption -->
		<video bind:this={video} src={url} playsinline controls autoplay onloadedmetadata={() => (video.playbackRate = speed)}
		></video>
		<div class="muted">{note}</div>
		{#if orders.length}
			<div class="row">
				{#each orders as order, i (i)}
					<button class="chip" onclick={() => jump(order.seconds)}
						>{clock(order.seconds)} {ORDERS[order.order] ?? order.order}</button
					>
				{/each}
			</div>
		{/if}
		<div class="bar">
			<span class="pills">
				{#each [1, 2, 4, 8] as x (x)}
					<button class:on={speed === x} onclick={() => (speed = x)}>{x}×</button>
				{/each}
			</span>
			<button class="chip" onclick={() => flag(null, game.game, video?.currentTime ?? 0, true)}
				>🚩 Flag</button
			>
		</div>
	</div>
</div>

<style>
	.backdrop {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.82);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 10;
		padding: env(safe-area-inset-top) 8px env(safe-area-inset-bottom);
	}
	.sheet {
		background: var(--bg);
		border: 1px solid var(--line);
		border-radius: 12px;
		width: 100%;
		max-width: 1100px;
		max-height: 100%;
		overflow-y: auto;
		padding: 10px;
	}
	.bar {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 8px;
		padding: 6px 2px;
	}
	video {
		width: 100%;
		max-height: 62vh;
		background: #000;
		display: block;
	}
	.pills {
		padding: 0;
	}
</style>
