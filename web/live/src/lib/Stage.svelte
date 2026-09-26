<script lang="ts">
	// The picture: the PC's stream while it streams; otherwise what its screen showed last,
	// dimmed, under what is going on (between games, the window closed, or no games at all).
	import Stream from './Stream.svelte';
	import { ago, day, lastLine } from './format';
	import { flag } from './api';
	import type { Station, Status } from './types';

	let { station, status }: { station: Station | null; status: Status } = $props();

	// The snapshot is asked for again once a minute at most.
	let minute = $state(Math.floor(Date.now() / 60000));
	$effect(() => {
		const timer = setInterval(() => (minute = Math.floor(Date.now() / 60000)), 15000);
		return () => clearInterval(timer);
	});

	const src = $derived(station?.streaming ? `s/${station.id}/live.m3u8` : null);
	const overlay = $derived.by(() => {
		if (!station) return { title: 'Connecting…', detail: '' };
		if (station.streaming) {
			if (station.game) return null;
			return { title: 'Between games', detail: station.last ? `Last: ${lastLine(station.last)}` : '' };
		}
		if (status.idle_since)
			return {
				title: 'No games running',
				detail: `The last one ended at ${day(status.idle_since)} (${ago(status.idle_since)}).`
			};
		return {
			title: station.id === 'here' ? 'No game on this PC' : 'Game window closed',
			detail: status.running.length
				? 'A recorder is running: the next game is on its way.'
				: station.last
					? `Last: ${lastLine(station.last)}`
					: ''
		};
	});
</script>

<div class="stage">
	{#if src}
		<Stream {src} />
	{:else if station}
		<img src={`s/${station.id}/latest.jpg?${minute}`} alt="" />
	{/if}
	{#if overlay}
		<div class="overlay">
			<div class="title">{overlay.title}</div>
			<div class="detail">{overlay.detail}</div>
		</div>
	{/if}
	{#if station?.game}
		{@const game = station.game}
		<button class="chip flag" onclick={() => flag(station.id, game.game, game.elapsed ?? 0, false)}
			>🚩 Flag</button
		>
	{/if}
</div>

<style>
	.stage {
		position: relative;
		background: #000;
		min-height: 180px;
	}
	img {
		display: block;
		width: 100%;
		aspect-ratio: 16 / 9;
		max-height: 72vh;
		object-fit: contain;
		opacity: 0.35;
	}
	.overlay {
		position: absolute;
		inset: 0;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		text-align: center;
		padding: 16px;
		pointer-events: none;
	}
	.title {
		font-size: 22px;
		font-weight: 700;
		text-shadow: 0 1px 6px #000;
	}
	.detail {
		margin-top: 6px;
		color: #d0d4d8;
		text-shadow: 0 1px 6px #000;
	}
	.flag {
		position: absolute;
		right: 10px;
		bottom: 54px;
		background: rgba(20, 22, 26, 0.8);
	}
</style>
