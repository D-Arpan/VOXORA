"use client";

import { useEffect, useRef } from "react";

class Particle {
    x: number;
    y: number;
    vx: number;
    vy: number;
    life: number;
    maxLife: number;
    size: number;
    color: string;

    constructor(x: number, y: number, isDarkMode: boolean) {
        this.x = x + (Math.random() - 0.5) * 40;
        this.y = y + (Math.random() - 0.5) * 40;
        this.vx = (Math.random() - 0.5) * 0.6;
        this.vy = (Math.random() - 0.5) * 0.6;
        this.maxLife = Math.random() * 80 + 40;
        this.life = this.maxLife;
        this.size = Math.random() * 3 + 1;
        
        // Cream/Ivory mode -> gold/teal glitter. Midnight mode -> emerald/cyan glitter.
        const hue = isDarkMode ? (Math.random() > 0.5 ? 150 : 180) : (Math.random() > 0.5 ? 45 : 170);
        const sat = 80 + Math.random() * 20;
        const lit = isDarkMode ? 60 + Math.random() * 20 : 40 + Math.random() * 20;
        this.color = `hsl(${hue}, ${sat}%, ${lit}%)`;
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.vy += 0.005; // Very slight gravity so they float smoothly
        this.life--;
    }

    draw(ctx: CanvasRenderingContext2D) {
        const alpha = Math.max(0, this.life / this.maxLife);
        ctx.globalAlpha = alpha;
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
}

export default function MouseGlitter({ isDarkMode }: { isDarkMode: boolean }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d", { alpha: true });
        if (!ctx) return;

        let particles: Particle[] = [];
        let animationFrameId: number;

        const resize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };
        window.addEventListener("resize", resize);
        resize();

        let lastSpawn = 0;
        const handleMouseMove = (e: MouseEvent) => {
            const now = Date.now();
            if (now - lastSpawn > 30) { // throttle spawn rate heavily
                for (let i = 0; i < 2; i++) { // spawn fewer particles
                    particles.push(new Particle(e.clientX, e.clientY, isDarkMode));
                }
                lastSpawn = now;
            }
        };
        window.addEventListener("mousemove", handleMouseMove);

        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            for (let i = particles.length - 1; i >= 0; i--) {
                const p = particles[i];
                p.update();
                p.draw(ctx);
                if (p.life <= 0) {
                    particles.splice(i, 1);
                }
            }
            animationFrameId = requestAnimationFrame(animate);
        };
        animate();

        return () => {
            window.removeEventListener("resize", resize);
            window.removeEventListener("mousemove", handleMouseMove);
            cancelAnimationFrame(animationFrameId);
        };
    }, [isDarkMode]);

    return (
        <canvas
            ref={canvasRef}
            className="fixed inset-0 pointer-events-none z-0"
            style={{ opacity: 0.8 }}
        />
    );
}
