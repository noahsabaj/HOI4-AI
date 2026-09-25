<script lang="ts">
	// What a game in progress is: the map and side, since when, the plan, how each side
	// stands in the game, and the latest orders; or the PC's last game between games.
	import { MILESTONES, ORDERS, SIDE, arena, clock, gameDate, hhmm, lastLine, planTags } from './format';
	import type { Side, Station } from './types';

	let { station }: { station: Station | null } = $props();
	const game = $derived(station?.game ?? null);
	const sides: Side[] = ['BLU', 'RED'];
</script>

<div class="card">
	{#if game}
		<div class="title">{arena(game.arena)} <span class={game.side}>{SIDE[game.side] ?? game.side}</span></div>
		<div class="muted">
			{[`Started ${hhmm(game.started_unix)}`, `${clock(game.elapsed)} in`, gameDate(game.date), game.run]
				.filter(Boolean)
				.join(' · ')}
		</div>
		<div class="row">
			{#each planTags(game.plan) as tag (tag)}<span class="tag">{tag}</span>{/each}
			{#if game.kicks}<span class="tag">camera kicks {game.kicks}</span>{/if}
		</div>
		{#if Object.keys(game.sides ?? {}).length}
			<div class="sides">
				{#each sides as side (side)}
					{@const r = game.sides[side]}
					{#if r}
						{@const pct = Math.round(100 * (r.surrender ?? 0))}
						<div class="side">
							<span class={side}>{SIDE[side]}{side === game.side ? ' (us)' : ''}</span>
							<div class="bar"><div class="fill fill-{side}" style:width="{pct}%"></div></div>
							<span class="muted"
								>{[
									`${pct}% surrender`,
									`${r.owned ?? '?'}/${r.states ?? '?'} states`,
									`${r.divisions ?? '?'} div`,
									r.strength != null ? `str ${r.strength.toFixed(2)}` : ''
								]
									.filter(Boolean)
									.join(' · ')}</span
							>
						</div>
					{/if}
				{/each}
			</div>
		{/if}
		{#if game.milestones}
			<div class="row">
				{#each Object.entries(MILESTONES) as [key, name] (key)}
					<span class="tag" class:win={game.milestones[key]}>{name} {game.milestones[key] ?? 0}</span>
				{/each}
			</div>
		{/if}
		{#if game.orders?.length}
			<ol class="orders">
				{#each game.orders.slice(-8).reverse() as order (`${order.frame}-${order.order}`)}
					<li>
						<time>{clock(order.seconds)}</time>{ORDERS[order.order] ?? order.order}{order.attack
							? ` (${order.attack})`
							: ''}
					</li>
				{/each}
			</ol>
		{/if}
	{:else if station?.last}
		<div class="muted">Last: {lastLine(station.last)}</div>
	{/if}
</div>

<style>
	.card {
		padding: 12px 16px 8px;
	}
	.title {
		font-size: 17px;
		font-weight: 650;
	}
	.sides {
		margin-top: 10px;
		display: grid;
		gap: 6px;
	}
	.side {
		display: grid;
		grid-template-columns: 70px 1fr auto;
		gap: 8px;
		align-items: center;
		font-size: 13px;
	}
	.bar {
		height: 8px;
		background: var(--panel);
		border-radius: 4px;
		overflow: hidden;
	}
	.fill {
		height: 100%;
		border-radius: 4px;
	}
	.fill-BLU {
		background: var(--blue);
	}
	.fill-RED {
		background: var(--red);
	}
	.orders {
		margin: 10px 0 0;
		padding: 0;
		list-style: none;
		font-size: 13px;
	}
	.orders li {
		padding: 2px 0;
		border-bottom: 1px solid var(--line);
	}
</style>
