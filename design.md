# Quantum Digital Twin Platform - Design System Prompt

## Design Vision

Create a **premium, immersive, and intelligent** web experience that feels like stepping into the future. The platform should communicate cutting-edge technology while remaining approachable and human-centered.

**Core Design Pillars:**
1. **Immersive** - Users feel like they're entering a quantum simulation sandbox
2. **Intelligent** - The interface feels alive, responsive, anticipatory
3. **Premium** - Enterprise-grade credibility with consumer-grade polish
4. **Clear** - Complex technology made visually intuitive

---

## Design References

### 1. Microsoft AI (microsoft.ai)
**What to take:**
- Clean, editorial typography with italicized emphasis words
- Human-centric imagery mixed with abstract visuals
- Light, airy backgrounds that breathe
- Card-based content layouts for news/updates
- "Approachable Intelligence" messaging tone
- Simple, confident navigation
- Professional but warm personality

### 2. Organimo (organimo.com)
**What to take:**
- Full-screen, scroll-driven storytelling
- Immersive journey metaphor ("scroll to begin")
- Creative typography with emphasized letters (Li*m*itless, T*h*e real)
- Smooth scroll-triggered animations
- Loading experience as part of the journey
- Expanding/collapsing benefit sections
- Audio toggle for ambient experience (optional)
- Organic transitions between sections

### 3. Mont-Fort Trading (mont-fort.com)
**What to take:**
- 3D page transitions and WebGL elements
- Premium corporate feel with depth
- Blue (#29648e) and light grey (#f4f6f8) color foundation
- Sophisticated scrolling interactions
- Gesture-based interactions
- Clean business credibility
- Award-winning animation quality (Awwwards SOTD)

---

## Color System

### Primary Palette

```
Quantum Blue       #1a365d    - Deep, trustworthy, primary actions
Electric Cyan      #00d4ff    - Quantum energy, highlights, accents
Neural Purple      #7c3aed    - AI/intelligence indicators
Success Green      #10b981    - Positive results, quantum advantage
```

### Neutral Palette

```
Void Black         #0a0a0f    - Deep backgrounds, text
Space Grey         #1f2937    - Secondary backgrounds
Mist               #f4f6f8    - Light backgrounds (from Mont-Fort)
Pure White         #ffffff    - Cards, content areas
```

### Semantic Colors

```
Quantum Advantage  #10b981    - When quantum beats classical
Classical Baseline #6b7280    - Classical comparison elements
Warning            #f59e0b    - Caution states
Error              #ef4444    - Error states
```

### Gradient System

```css
/* Primary quantum gradient - use for hero, key CTAs */
.quantum-gradient {
  background: linear-gradient(135deg, #1a365d 0%, #7c3aed 50%, #00d4ff 100%);
}

/* Subtle glow effect for quantum elements */
.quantum-glow {
  box-shadow: 0 0 60px rgba(0, 212, 255, 0.3);
}

/* Dark immersive background */
.void-gradient {
  background: radial-gradient(ellipse at center, #1f2937 0%, #0a0a0f 100%);
}
```

---

## Typography

### Font Stack

```css
/* Primary - Clean, modern, technical */
--font-primary: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;

/* Display - For hero headlines, impact moments */
--font-display: 'Space Grotesk', sans-serif;

/* Mono - For code, quantum metrics, technical data */
--font-mono: 'JetBrains Mono', 'Fira Code', monospace;
```

### Type Scale

```css
/* Following Microsoft AI's editorial approach */
--text-hero: clamp(3rem, 8vw, 7rem);      /* Hero headlines */
--text-h1: clamp(2.5rem, 5vw, 4rem);      /* Page titles */
--text-h2: clamp(1.75rem, 3vw, 2.5rem);   /* Section headers */
--text-h3: clamp(1.25rem, 2vw, 1.75rem);  /* Subsections */
--text-body: 1.125rem;                     /* Body copy */
--text-small: 0.875rem;                    /* Captions, labels */
--text-micro: 0.75rem;                     /* Technical data */
```

### Typography Patterns

**Hero Headlines (Microsoft AI style):**
```html
<h1>
  Build a <em>Second</em> World
</h1>
```

**Emphasized Letters (Organimo style):**
```html
<h2>
  Q<span class="highlight">u</span>antum 
  Adv<span class="highlight">a</span>ntage
</h2>
```

---

## Layout System

### Grid

```css
/* 12-column fluid grid */
.container {
  max-width: 1440px;
  margin: 0 auto;
  padding: 0 clamp(1rem, 5vw, 4rem);
}

/* Content width for readability */
.content-width {
  max-width: 720px;
}

/* Wide content for dashboards */
.wide-width {
  max-width: 1200px;
}
```

### Spacing Scale

```css
--space-1: 0.25rem;   /* 4px */
--space-2: 0.5rem;    /* 8px */
--space-3: 1rem;      /* 16px */
--space-4: 1.5rem;    /* 24px */
--space-5: 2rem;      /* 32px */
--space-6: 3rem;      /* 48px */
--space-7: 4rem;      /* 64px */
--space-8: 6rem;      /* 96px */
--space-9: 8rem;      /* 128px */
--space-10: 12rem;    /* 192px */
```

### Section Heights

```css
/* Full viewport immersive sections (Organimo style) */
.section-full {
  min-height: 100vh;
  min-height: 100dvh; /* Dynamic viewport height */
}

/* Standard content sections */
.section-standard {
  padding: var(--space-9) 0;
}
```

---

## Component Library

### Navigation

**Style:** Minimal, floating, glass-morphism

```
┌─────────────────────────────────────────────────────────────────┐
│  ⚛️ Quantum Twin          Builder   Showcase   Docs      [Login]│
│                                                                  │
│  (Glass background with subtle blur, appears on scroll)         │
└─────────────────────────────────────────────────────────────────┘
```

**Behavior:**
- Transparent on hero, glass-morphism on scroll
- Sticky with smooth transition
- Mobile: Hamburger with full-screen overlay

### Hero Section

**Style:** Full-viewport, immersive, scroll-triggered (Organimo + Mont-Fort)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│                                                                  │
│                   Build a Second World                           │
│                                                                  │
│          Describe any system. Simulate infinite futures.         │
│                    Powered by quantum.                           │
│                                                                  │
│                    [ Start Building ]                            │
│                                                                  │
│                         ↓ scroll                                 │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │     (3D quantum particle simulation - WebGL)             │    │
│  │     Particles form shapes based on scroll position       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Elements:**
- Animated particle system (Three.js/WebGL)
- Particles respond to mouse movement
- Text reveals on scroll
- Subtle ambient audio toggle

### Conversation Interface

**Style:** Clean, focused, AI-forward

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ⚛️ What system would you like to simulate?              │    │
│  │                                                          │    │
│  │ I can help you build a quantum digital twin for any     │    │
│  │ domain - healthcare, logistics, finance, athletics,     │    │
│  │ military operations, ecosystems, or anything else       │    │
│  │ you can describe.                                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 👤 I run a logistics company with 50 vehicles across    │    │
│  │    12 cities. I want to optimize our delivery routes    │    │
│  │    while minimizing fuel costs.                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ⚛️ Great! I can build a Quantum Logistics Twin for you. │    │
│  │                                                          │    │
│  │ ┌─────────────────────────────────────────────────────┐ │    │
│  │ │ 📊 UNDERSTANDING YOUR SYSTEM                        │ │    │
│  │ │                                                      │ │    │
│  │ │ Entities: 50 vehicles, 12 cities                    │ │    │
│  │ │ Objective: Minimize fuel + optimize routes          │ │    │
│  │ │ Problem type: Combinatorial Optimization            │ │    │
│  │ │ Quantum algorithm: QAOA                             │ │    │
│  │ └─────────────────────────────────────────────────────┘ │    │
│  │                                                          │    │
│  │ I need a bit more information...                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Type your message...                            [Send ➤]│    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  [📎 Upload Data]  [🎤 Voice]  [💡 Examples]                    │
└─────────────────────────────────────────────────────────────────┘
```

**Features:**
- Typing indicator with quantum-style animation
- Inline system understanding cards
- Smooth message transitions
- File upload with drag-and-drop
- Code/data syntax highlighting

### Twin Generation Progress

**Style:** Immersive, educational (Organimo journey style)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│              Generating Your Quantum Twin                        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                                                          │    │
│  │         (3D visualization of quantum circuit             │    │
│  │          being constructed - animated WebGL)             │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━░░░░░░░░░░  67%             │
│                                                                  │
│  ✓ System extracted                                              │
│  ✓ Entities mapped to qubits (50 → 6 qubits)                    │
│  ✓ Constraints encoded                                           │
│  ◉ Building QAOA circuit...                                      │
│  ○ Optimizing parameters                                         │
│  ○ Validating twin                                               │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 💡 While you wait...                                     │    │
│  │                                                          │    │
│  │ QAOA (Quantum Approximate Optimization Algorithm)       │    │
│  │ tests all possible route combinations simultaneously    │    │
│  │ using quantum superposition. Classical algorithms       │    │
│  │ would need to test them one by one.                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Interactive Dashboard

**Style:** Data-rich but clean, simulation controls prominent

```
┌─────────────────────────────────────────────────────────────────┐
│  ← Back    Logistics Twin    [⚡ Quantum Active]    [⚙️] [📤]   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐│
│  │ Efficiency  │ │ Fuel Saved  │ │ Routes      │ │ Quantum    ││
│  │             │ │             │ │ Optimized   │ │ Advantage  ││
│  │   87%       │ │  $12.4K     │ │    47       │ │   340x     ││
│  │   ↑ 23%     │ │  /month     │ │   /50       │ │  faster    ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘│
│                                                                  │
│  ┌─────────────────────────────────┬───────────────────────────┐│
│  │                                  │                           ││
│  │   (Interactive route map)       │  SIMULATION CONTROLS      ││
│  │                                  │                           ││
│  │   Vehicles shown as animated    │  Time: [|◀][◀◀][▶][▶▶][▶|]││
│  │   dots following routes         │                           ││
│  │                                  │  Speed: [━━━━━○━━━━━]     ││
│  │   Click vehicle for details     │                           ││
│  │                                  │  ┌─────────────────────┐ ││
│  │                                  │  │ 🔀 What if...       │ ││
│  │                                  │  │                     │ ││
│  │                                  │  │ [Add 5 vehicles   ] │ ││
│  │                                  │  │ [Remove route 7   ] │ ││
│  │                                  │  │ [Double demand    ] │ ││
│  │                                  │  │ [Custom scenario  ] │ ││
│  │                                  │  └─────────────────────┘ ││
│  └─────────────────────────────────┴───────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 💬 Ask your twin...                                   [Ask] ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Quantum Advantage Showcase (Healthcare)

**Style:** Educational, comparative, interactive (Mont-Fort depth + Microsoft clarity)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│                How Quantum Beats Classical                       │
│                                                                  │
│          A deep dive into our healthcare case study              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  [Overview] [Implementation] [Benchmarks] [Try It Live]    │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │                         │  │                              │  │
│  │   CLASSICAL             │  │   QUANTUM                    │  │
│  │   Genetic Algorithm     │  │   QAOA                       │  │
│  │                         │  │                              │  │
│  │   ████████░░░░ 67%      │  │   ████████████████ 100%      │  │
│  │   Time: 4.2s            │  │   Time: 0.3s                 │  │
│  │   Tested: 120 combos    │  │   Tested: ALL combos         │  │
│  │   Result: Local optimum │  │   Result: Global optimum     │  │
│  │                         │  │                              │  │
│  │   [View Algorithm]      │  │   [View Circuit]             │  │
│  │                         │  │                              │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
│                                                                  │
│                     ⚡ 14x faster + guaranteed optimal           │
│                                                                  │
│             [ Run Your Own Comparison ]                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Benchmark Results Display

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│                    Validated Results                             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Module               Classical    Quantum     Advantage    │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │ Personalized Med.    1K/hr        1M/hr       ████ 1000x   │ │
│  │ Drug Discovery       1000 hrs     1 hr        ████ 1000x   │ │
│  │ Medical Imaging      74%          87%         ██ +13%      │ │
│  │ Genomic Analysis     100 genes    1000+       ████ 10x     │ │
│  │ Epidemic Modeling    3 days       6 min       ████ 720x    │ │
│  │ Hospital Ops         baseline     -73% wait   ███ 73%      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Statistical Validation                                          │
│  ───────────────────────                                         │
│  Overall Accuracy: 85%  │  Sensitivity: 90%  │  p < 0.001       │
│                                                                  │
│             [ View Methodology ]  [ Reproduce Tests ]            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Animation System

### Principles

1. **Purposeful** - Every animation communicates something
2. **Quantum-Themed** - Particles, waves, superposition visualizations
3. **Performant** - 60fps, GPU-accelerated
4. **Interruptible** - User actions take priority

### Timing

```css
--ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1);
--ease-in-out-sine: cubic-bezier(0.37, 0, 0.63, 1);

--duration-instant: 100ms;
--duration-fast: 200ms;
--duration-normal: 300ms;
--duration-slow: 500ms;
--duration-glacial: 1000ms;
```

### Key Animations

**Page Transitions (Mont-Fort style):**
- 3D depth transitions between major sections
- Content slides with parallax layers
- WebGL particle system morphs

**Scroll Animations (Organimo style):**
- Elements reveal as user scrolls
- Parallax depth on backgrounds
- Text characters animate in sequence
- Progress indicators

**Quantum Visualizations:**
- Particle systems representing data
- Wave function collapse on decisions
- Entanglement lines connecting related elements
- Superposition shimmer effect

**Micro-interactions:**
- Button hover: subtle glow + lift
- Card hover: depth increase + content preview
- Input focus: border animation + label float
- Loading: quantum particle orbit

### Loading States

**Full Page (Organimo style):**
```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│                                                                  │
│                    (Quantum particle animation)                  │
│                                                                  │
│                         Loading...                               │
│                                                                  │
│                    ━━━━━━━━━━━━━━━━━━━━                          │
│                                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Inline (for queries, generation):**
```
⚛️ ○ ○ ○  (orbiting particles)
```

---

## Responsive Design

### Breakpoints

```css
--bp-mobile: 480px;
--bp-tablet: 768px;
--bp-laptop: 1024px;
--bp-desktop: 1280px;
--bp-wide: 1536px;
```

### Mobile Considerations

- Full-screen sections maintained
- Simplified 3D (reduce particle count)
- Bottom navigation for dashboard
- Swipe gestures for simulation controls
- Conversation interface optimized for thumb reach
- Charts become horizontally scrollable

### Touch Interactions

- Swipe between dashboard views
- Pinch-zoom on visualizations
- Long-press for details
- Pull-to-refresh for data

---

## WebGL / 3D Elements

### Hero Particle System

```javascript
// Concept: Particles that form quantum circuit patterns
// - Respond to mouse/scroll
// - Form shapes (atoms, waves, circuits)
// - Color shifts based on section
// Libraries: Three.js, custom shaders
```

### Circuit Visualization

```javascript
// Show quantum circuits being built
// - Gates appear as nodes
// - Connections animate
// - Measurement collapses state
// Libraries: Three.js or D3.js
```

### Data Twin Visualization

```javascript
// 3D representation of the user's system
// - Entities as nodes
// - Relationships as edges
// - States as colors/sizes
// Libraries: Three.js force-directed graph
```

### Performance Guidelines

- Lazy-load WebGL scenes
- Reduce complexity on mobile
- Provide fallback static images
- Use requestAnimationFrame
- Pause when off-screen

---

## Accessibility

### Requirements

- WCAG 2.1 AA compliance minimum
- Keyboard navigation for all features
- Screen reader support
- Reduced motion option
- High contrast mode
- Focus indicators

### Implementation

```css
/* Respect user preferences */
@media (prefers-reduced-motion: reduce) {
  * {
    animation: none !important;
    transition: none !important;
  }
}

@media (prefers-color-scheme: dark) {
  /* Already dark by default, ensure contrast */
}

@media (prefers-contrast: high) {
  /* Increase contrast, simplify gradients */
}
```

---

## Voice & Tone

### Personality

- **Confident** but not arrogant
- **Technical** but accessible
- **Visionary** but grounded
- **Warm** but professional

### Messaging Examples

**Hero:**
> "Build a Second World"
> "Describe any reality. Simulate infinite futures."

**Feature Introduction:**
> "Your quantum twin is ready. Ask it anything."

**Quantum Explanation:**
> "While classical computers test one scenario at a time, quantum computers test millions simultaneously. That's not hyperbole—it's physics."

**Error State:**
> "Something went wrong. Let's try that again."

**Success:**
> "Your twin is live. The quantum advantage begins now."

---

## Dark/Light Mode

### Default: Dark

The platform defaults to dark mode to:
- Emphasize the "quantum void" / simulation aesthetic
- Reduce eye strain during long sessions
- Make data visualizations pop
- Feel more immersive

### Light Mode (Optional)

For users who prefer it:
- Swap void black → mist white
- Adjust quantum blue for contrast
- Maintain all functionality

---

## Implementation Notes

### Technology Recommendations

**Framework:** Next.js 14+ (App Router)
**Styling:** Tailwind CSS + CSS Modules for complex components
**Animation:** Framer Motion + GSAP for scroll animations
**3D:** Three.js with React Three Fiber
**Charts:** Recharts or Plotly for data viz
**Icons:** Lucide React

### File Structure

```
/app
  /builder          # Universal Twin Builder
  /showcase         # Quantum Advantage Showcase
    /healthcare
      /[module]     # Individual module comparisons
  /dashboard        # Twin dashboards
  /api              # Backend routes

/components
  /ui               # Base components (buttons, inputs)
  /layout           # Navigation, containers
  /conversation     # Chat interface
  /visualization    # Charts, 3D, animations
  /showcase         # Benchmark displays

/lib
  /three            # WebGL scenes
  /animations       # Animation configs
  /hooks            # Custom hooks
```

---

## Summary: The Feel

When a user lands on this platform, they should feel like they're:

1. **Entering a sandbox for reality** - The hero draws them in with particles and depth
2. **Talking to an intelligence** - The conversation interface feels alive
3. **Watching something being built** - The generation progress is mesmerizing
4. **Controlling a simulation** - The dashboard puts power in their hands
5. **Understanding the magic** - The showcase proves it's not just hype

**Design is the proof that quantum is accessible.**

---

*Immersive. Intelligent. Infinite possibilities.*