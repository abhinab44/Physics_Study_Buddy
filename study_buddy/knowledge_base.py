# Domain: Study Buddy — Physics
# Extensive Physics Knowledge Base for B.Tech Students
# 32 course-aligned documents covering the complete Engineering Physics syllabus
# Each document contains formulas, theorems, definitions, and key concepts

import chromadb
from sentence_transformers import SentenceTransformer

DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Units and Measurement",
        "text": (
            "Unit 1: Units and Measurement. "
            "A unit is a standard quantity used for measurement. The SI system (Systeme International) "
            "has seven fundamental (base) units: metre (m) for length, kilogram (kg) for mass, "
            "second (s) for time, ampere (A) for electric current, kelvin (K) for temperature, "
            "mole (mol) for amount of substance, and candela (cd) for luminous intensity. "
            "Derived units are combinations of fundamental units, e.g. velocity (m/s), force (kg m/s^2 = Newton), "
            "energy (kg m^2/s^2 = Joule). "
            "Dimensions express a physical quantity in terms of fundamental quantities using symbols: "
            "[M] for mass, [L] for length, [T] for time, [A] for current, [K] for temperature. "
            "For example, Force has dimensions [M L T^-2], Energy has dimensions [M L^2 T^-2], "
            "Power has dimensions [M L^2 T^-3]. "
            "Applications of dimensional analysis: (1) checking dimensional consistency of equations, "
            "(2) deriving relations between physical quantities, (3) converting units between systems. "
            "Limitations: dimensional analysis cannot determine dimensionless constants, cannot distinguish "
            "between quantities with the same dimensions (e.g. work and torque both have [M L^2 T^-2]), "
            "and cannot derive equations involving trigonometric, exponential, or logarithmic functions. "
            "Significant figures indicate the precision of a measurement. Rules: all non-zero digits are "
            "significant, zeros between non-zero digits are significant, trailing zeros after a decimal "
            "point are significant."
        ),
    },
    {
        "id": "doc_002",
        "topic": "Motion in One Dimension",
        "text": (
            "Unit 2: Description of Motion in One Dimension. "
            "Motion in a straight line is described using position, displacement, velocity, and acceleration. "
            "Displacement is the change in position vector: Delta_x = x_final - x_initial (a vector quantity). "
            "Distance is the total path length (scalar, always positive). "
            "Average velocity: v_avg = Delta_x / Delta_t. Instantaneous velocity: v = dx/dt (derivative of position). "
            "Average acceleration: a_avg = Delta_v / Delta_t. Instantaneous acceleration: a = dv/dt = d^2x/dt^2. "
            "Equations of uniformly accelerated motion (SUVAT equations): "
            "(1) v = u + at, (2) s = ut + (1/2)at^2, (3) v^2 = u^2 + 2as, "
            "(4) s = (u + v)t/2, (5) s_nth = u + a(2n - 1)/2 (displacement in nth second). "
            "Here u = initial velocity, v = final velocity, a = acceleration, t = time, s = displacement. "
            "Graphical representation: In a position-time (x-t) graph, slope = velocity. "
            "In a velocity-time (v-t) graph, slope = acceleration, and area under the curve = displacement. "
            "In an acceleration-time (a-t) graph, area under curve = change in velocity. "
            "Free fall: motion under gravity alone with a = g = 9.8 m/s^2 downward. "
            "For an object thrown vertically upward: v = u - gt, h = ut - (1/2)gt^2. "
            "Maximum height: H = u^2/(2g). Time of ascent = u/g. Time of flight = 2u/g."
        ),
    },
    {
        "id": "doc_003",
        "topic": "Motion in Two and Three Dimensions",
        "text": (
            "Unit 3: Description of Motion in Two and Three Dimensions. "
            "Scalars have only magnitude (mass, temperature, speed). Vectors have magnitude and direction "
            "(displacement, velocity, force). "
            "Vector addition: Triangle law and parallelogram law. "
            "If two vectors A and B make angle theta, resultant R = sqrt(A^2 + B^2 + 2AB cos(theta)). "
            "Direction: tan(alpha) = B sin(theta) / (A + B cos(theta)). "
            "A zero vector (null vector) has zero magnitude and arbitrary direction. Properties: "
            "A + 0 = A, A - A = 0. "
            "Resolution of vectors: Any vector can be resolved into components along perpendicular axes. "
            "A = Ax i_hat + Ay j_hat + Az k_hat where Ax = A cos(alpha), Ay = A cos(beta), Az = A cos(gamma). "
            "Scalar (dot) product: A . B = AB cos(theta). Properties: commutative, distributive, "
            "i.i = j.j = k.k = 1, i.j = j.k = k.i = 0. If A . B = 0, vectors are perpendicular. "
            "Vector (cross) product: A x B = AB sin(theta) n_hat. Properties: anti-commutative (A x B = -B x A), "
            "i x j = k, j x k = i, k x i = j, i x i = j x j = k x k = 0. "
            "Projectile motion: A projectile launched at angle theta with velocity u has: "
            "Horizontal range R = u^2 sin(2*theta)/g. Maximum height H = u^2 sin^2(theta)/(2g). "
            "Time of flight T = 2u sin(theta)/g. Maximum range at theta = 45 degrees: R_max = u^2/g. "
            "Uniform circular motion: speed is constant but velocity changes direction continuously. "
            "Centripetal acceleration a_c = v^2/r = omega^2 * r directed toward the center. "
            "Angular velocity omega = 2*pi/T = 2*pi*f where T = period, f = frequency."
        ),
    },
    {
        "id": "doc_004",
        "topic": "Laws of Motion",
        "text": (
            "Unit 4: Laws of Motion. "
            "Newton's First Law (Law of Inertia): A body remains at rest or in uniform motion in a straight "
            "line unless acted upon by an external unbalanced force. Inertia is the tendency of a body to "
            "resist changes in its state of motion. Mass is the measure of inertia. "
            "Newton's Second Law: The rate of change of momentum is proportional to the applied force. "
            "F = dp/dt = m * a (for constant mass). Force is measured in Newtons: 1 N = 1 kg m/s^2. "
            "Momentum: p = m * v (vector). Impulse: J = F * Delta_t = Delta_p (change in momentum). "
            "Newton's Third Law: For every action, there is an equal and opposite reaction. "
            "Action and reaction act on DIFFERENT bodies simultaneously. "
            "Conservation of Linear Momentum: In the absence of external forces, total momentum of a "
            "system remains constant. m1*u1 + m2*u2 = m1*v1 + m2*v2. "
            "Rocket propulsion: The rocket expels gas backward (action), and the rocket moves forward (reaction). "
            "Thrust F = v_rel * (dm/dt) where v_rel is exhaust velocity and dm/dt is rate of mass ejection. "
            "Rocket equation: v = u + v_rel * ln(M0/M) where M0 is initial mass, M is current mass. "
            "Friction: the force that opposes relative motion between surfaces. "
            "Static friction: f_s <= mu_s * N (up to a maximum). Kinetic friction: f_k = mu_k * N. "
            "mu_s > mu_k always. Laws of friction: (1) friction is independent of area of contact, "
            "(2) friction is proportional to normal reaction, (3) kinetic friction is independent of velocity. "
            "Angle of friction: tan(phi) = mu. Angle of repose: tan(theta) = mu_s."
        ),
    },
    {
        "id": "doc_005",
        "topic": "Work, Energy and Power",
        "text": (
            "Unit 5: Work, Energy and Power. "
            "Work done by a constant force: W = F . d = F * d * cos(theta). "
            "If theta = 0, W = Fd (maximum work). If theta = 90, W = 0 (no work). If theta > 90, W < 0 (negative work). "
            "Work done by a variable force: W = integral of F . dx from x1 to x2. "
            "Kinetic Energy: KE = (1/2) * m * v^2. It is the energy of motion (always positive). "
            "Work-Energy Theorem: Net work done = change in kinetic energy. W_net = KE_f - KE_i = (1/2)m(v^2 - u^2). "
            "Potential Energy: energy due to position or configuration. "
            "Gravitational PE: U = mgh (near Earth's surface). Elastic PE: U = (1/2)kx^2 (for a spring). "
            "Conservative forces: work done is path-independent (gravity, spring force, electrostatic). "
            "Non-conservative forces: work done is path-dependent (friction, air resistance). "
            "Conservation of Energy: Total energy (KE + PE) remains constant for conservative forces. "
            "(1/2)mv^2 + mgh = constant. Energy can be transformed but not created or destroyed. "
            "Power: rate of doing work. P = W/t = F . v. Unit: Watt (1 W = 1 J/s). 1 HP = 746 W. "
            "Elastic collision: both momentum and kinetic energy are conserved. "
            "In 1D elastic collision: v1 = ((m1-m2)/(m1+m2))*u1 + (2m2/(m1+m2))*u2. "
            "Inelastic collision: momentum is conserved but kinetic energy is NOT. "
            "Perfectly inelastic: bodies stick together. v = (m1*u1 + m2*u2)/(m1 + m2). "
            "Coefficient of restitution: e = (v2 - v1)/(u1 - u2). e = 1 for elastic, e = 0 for perfectly inelastic. "
            "In 2D elastic collision, momentum is conserved component-wise along x and y axes."
        ),
    },
    {
        "id": "doc_006",
        "topic": "Rotational Motion and Moment of Inertia",
        "text": (
            "Unit 6: Rotational Motion and Moment of Inertia. "
            "Centre of mass of a two-particle system: x_cm = (m1*x1 + m2*x2)/(m1 + m2). "
            "For a rigid body: R_cm = (1/M) * sum(m_i * r_i). The centre of mass of a uniform body "
            "lies at its geometric centre. "
            "General motion of a rigid body = translation of centre of mass + rotation about centre of mass. "
            "Torque (moment of force): tau = r x F. Magnitude: tau = r * F * sin(theta). Unit: N.m. "
            "Angular momentum: L = r x p = I * omega. Unit: kg m^2/s. "
            "Newton's second law for rotation: tau = I * alpha = dL/dt. "
            "Conservation of angular momentum: If net external torque is zero, L = I*omega = constant. "
            "Example: ice skater pulls arms in, I decreases, omega increases. "
            "Moment of inertia: I = sum(m_i * r_i^2) for discrete masses. I = integral(r^2 dm) for continuous. "
            "It depends on mass distribution relative to the axis of rotation. "
            "Parallel axis theorem: I = I_cm + M*d^2, where d = distance between parallel axes. "
            "Perpendicular axis theorem (for planar bodies): I_z = I_x + I_y. "
            "Standard moments of inertia: "
            "Thin ring (axis through centre, perpendicular): I = MR^2. "
            "Disc (axis through centre, perpendicular): I = (1/2)MR^2. "
            "Solid sphere (diameter): I = (2/5)MR^2. "
            "Hollow sphere (diameter): I = (2/3)MR^2. "
            "Thin rod (centre, perpendicular): I = (1/12)ML^2. "
            "Thin rod (end, perpendicular): I = (1/3)ML^2. "
            "Rotational KE = (1/2)I*omega^2. Rolling without slipping: v = R*omega. "
            "Total KE of rolling body = (1/2)mv^2 + (1/2)I*omega^2 = (1/2)mv^2(1 + k^2/R^2) "
            "where k = radius of gyration (I = Mk^2)."
        ),
    },
    {
        "id": "doc_007",
        "topic": "Gravitation",
        "text": (
            "Unit 7: Gravitation. "
            "Newton's Universal Law of Gravitation: F = G*m1*m2/r^2. "
            "G = 6.674 x 10^-11 N m^2/kg^2 (universal gravitational constant). "
            "Acceleration due to gravity at Earth's surface: g = GM/R^2 = 9.8 m/s^2. "
            "Variation of g: With height h: g' = g(1 - 2h/R) for h << R. "
            "With depth d: g' = g(1 - d/R). At centre of Earth (d = R): g' = 0. "
            "With latitude lambda: g' = g - R*omega^2*cos^2(lambda). g is maximum at poles, minimum at equator. "
            "Kepler's Laws of Planetary Motion: "
            "(1) Law of Orbits: Planets move in elliptical orbits with the Sun at one focus. "
            "(2) Law of Areas: The line joining a planet to the Sun sweeps equal areas in equal times. "
            "This implies angular momentum is conserved: L = m*v*r = constant. "
            "(3) Law of Periods: T^2 is proportional to a^3 (T^2 = (4*pi^2/(GM)) * a^3) where a = semi-major axis. "
            "Orbital velocity of satellite: v_o = sqrt(GM/r) = sqrt(g*R^2/r). "
            "For near-surface orbit: v_o = sqrt(gR) = 7.9 km/s. "
            "Time period of satellite: T = 2*pi*sqrt(r^3/(GM)). "
            "Geostationary satellite: T = 24 hours, orbits in equatorial plane, height = 35,786 km. "
            "Gravitational potential energy: U = -GMm/r (zero at infinity). "
            "Near surface: Delta_U = mgh. "
            "Gravitational potential: V = -GM/r (potential energy per unit mass). "
            "Escape velocity: v_e = sqrt(2GM/R) = sqrt(2gR) = 11.2 km/s for Earth. "
            "Relation: v_e = sqrt(2) * v_o. "
            "Binding energy = -E = GMm/(2r) for a satellite in orbit."
        ),
    },
    {
        "id": "doc_008",
        "topic": "Solids and Fluids",
        "text": (
            "Unit 8: Solids and Fluids. "
            "Inter-atomic forces: attractive at large distances, repulsive at very short distances. "
            "Equilibrium separation r_0 is where net force is zero (potential energy is minimum). "
            "States of matter: Solid (fixed shape, fixed volume), Liquid (no fixed shape, fixed volume), "
            "Gas (no fixed shape, no fixed volume). "
            "A) SOLIDS - Elastic Properties: "
            "Stress = Force/Area = F/A (unit: Pa = N/m^2). "
            "Strain = change in dimension / original dimension (dimensionless). "
            "Hooke's Law: Within elastic limit, stress is proportional to strain. stress = E * strain. "
            "Young's modulus: Y = (F/A) / (Delta_L/L) = (F*L)/(A*Delta_L). Measures resistance to elongation. "
            "Bulk modulus: B = -V*(Delta_P/Delta_V) = -Delta_P/(Delta_V/V). Measures resistance to compression. "
            "Modulus of rigidity (shear modulus): G = shear stress / shear strain = (F/A) / tan(phi). "
            "Poisson's ratio: sigma = lateral strain / longitudinal strain (typically 0 to 0.5). "
            "B) LIQUIDS: "
            "Cohesion: force between molecules of the same substance. "
            "Adhesion: force between molecules of different substances. "
            "Surface tension: S = F/L (force per unit length). Unit: N/m. "
            "Surface energy: E = S * Delta_A (surface tension times change in area). "
            "Pressure inside a soap bubble: Delta_P = 4S/R (two surfaces). "
            "Pressure inside a liquid drop: Delta_P = 2S/R (one surface). "
            "Capillary rise: h = 2S*cos(theta)/(rho*g*r). "
            "Bernoulli's theorem: P + (1/2)*rho*v^2 + rho*g*h = constant along a streamline. "
            "Applications: Venturi meter, atomizer, airplane lift, Magnus effect. "
            "Viscosity: F = -eta * A * (dv/dx) (Newton's law of viscosity). Unit of eta: Pa.s or Poise. "
            "Stoke's Law: F = 6*pi*eta*r*v (drag on a sphere). "
            "Terminal velocity: v_t = (2*r^2*(rho_s - rho_l)*g) / (9*eta)."
        ),
    },
    {
        "id": "doc_009",
        "topic": "Oscillations",
        "text": (
            "Unit 9: Oscillations. "
            "Periodic motion: motion that repeats after a fixed time interval (period T). "
            "Frequency f = 1/T. Angular frequency omega = 2*pi*f = 2*pi/T. "
            "Simple Harmonic Motion (SHM): restoring force is proportional to displacement from equilibrium "
            "and directed toward it. F = -kx (Hooke's law form). "
            "Equation of motion: m*(d^2x/dt^2) + kx = 0, or d^2x/dt^2 + omega^2*x = 0 where omega = sqrt(k/m). "
            "Solution: x(t) = A*sin(omega*t + phi) or x(t) = A*cos(omega*t + phi). "
            "A = amplitude, phi = initial phase, (omega*t + phi) = instantaneous phase. "
            "Velocity: v = dx/dt = A*omega*cos(omega*t + phi). Maximum velocity v_max = A*omega at mean position. "
            "Acceleration: a = -omega^2*x. Maximum acceleration a_max = A*omega^2 at extreme positions. "
            "At displacement x: v = omega*sqrt(A^2 - x^2). "
            "Energy in SHM: KE = (1/2)*m*omega^2*(A^2 - x^2). PE = (1/2)*m*omega^2*x^2. "
            "Total energy E = (1/2)*m*omega^2*A^2 = (1/2)*k*A^2 = constant. "
            "At mean position: KE = max, PE = 0. At extreme: KE = 0, PE = max. "
            "Spring oscillation: T = 2*pi*sqrt(m/k). "
            "Springs in series: 1/k_eff = 1/k1 + 1/k2. Springs in parallel: k_eff = k1 + k2. "
            "Simple pendulum: T = 2*pi*sqrt(L/g) (for small angles theta < 15 degrees). "
            "The period is independent of mass and amplitude (for small oscillations). "
            "Effective g in an elevator: going up with acceleration a, g_eff = g + a; "
            "going down, g_eff = g - a; free fall, g_eff = 0 (infinite period)."
        ),
    },
    {
        "id": "doc_010",
        "topic": "Waves",
        "text": (
            "Unit 10: Waves. "
            "A wave is a disturbance that transfers energy without transferring matter. "
            "Transverse waves: particles vibrate perpendicular to wave direction (light, strings). "
            "Longitudinal waves: particles vibrate parallel to wave direction (sound in air). "
            "Wave equation: y(x,t) = A*sin(kx - omega*t) for a progressive wave moving in +x direction. "
            "Wave number k = 2*pi/lambda. Wavelength lambda = distance for one complete cycle. "
            "Wave speed: v = f*lambda = omega/k. "
            "Speed of transverse wave on a string: v = sqrt(T/mu) where T = tension, mu = mass per unit length. "
            "Speed of sound in a medium: v = sqrt(B/rho) where B = bulk modulus, rho = density. "
            "Speed of sound in air at temperature T: v = 331.4*sqrt(T/273) m/s. "
            "Superposition principle: when two waves overlap, the resultant displacement is the sum of individual displacements. "
            "Standing waves: formed by superposition of two identical waves travelling in opposite directions. "
            "y = 2A*sin(kx)*cos(omega*t). Nodes: kx = n*pi, Antinodes: kx = (2n+1)*pi/2. "
            "String fixed at both ends: lambda_n = 2L/n, f_n = n*v/(2L). Fundamental frequency f_1 = v/(2L). "
            "Open pipe: f_n = n*v/(2L) (all harmonics). Closed pipe: f_n = n*v/(4L) (odd harmonics only, n = 1,3,5...). "
            "Beats: produced when two waves of slightly different frequencies interfere. "
            "Beat frequency = |f1 - f2|. y = 2A*cos(2*pi*(f1-f2)*t/2)*sin(2*pi*(f1+f2)*t/2). "
            "Doppler effect: apparent change in frequency due to relative motion. "
            "General formula: f' = f * (v + v_observer) / (v - v_source) "
            "(use + for motion toward each other, - for motion away). "
            "Resonance: when driving frequency matches natural frequency, amplitude becomes maximum."
        ),
    },
    {
        "id": "doc_011",
        "topic": "Heat and Thermodynamics",
        "text": (
            "Unit 11: Heat and Thermodynamics. "
            "Thermal expansion: Linear: Delta_L = L*alpha*Delta_T. Area: Delta_A = A*2*alpha*Delta_T. "
            "Volume: Delta_V = V*3*alpha*Delta_T (or V*gamma*Delta_T). "
            "alpha = coefficient of linear expansion. gamma = 3*alpha (coefficient of volume expansion). "
            "Specific heat capacity: c = Q/(m*Delta_T). Unit: J/(kg.K). "
            "Heat capacity: C = Q/Delta_T = m*c. "
            "For ideal gases: Cp - Cv = R (Mayer's relation). gamma = Cp/Cv. "
            "For monoatomic gas: Cv = (3/2)R, Cp = (5/2)R, gamma = 5/3. "
            "For diatomic gas: Cv = (5/2)R, Cp = (7/2)R, gamma = 7/5. "
            "First Law of Thermodynamics: Delta_U = Q - W (energy conservation). "
            "Q = heat added to system, W = work done by system, Delta_U = change in internal energy. "
            "For ideal gas: Delta_U = n*Cv*Delta_T. "
            "Thermodynamic processes: "
            "Isothermal (T = const): PV = const. W = nRT*ln(V2/V1). Delta_U = 0. "
            "Adiabatic (Q = 0): PV^gamma = const. TV^(gamma-1) = const. W = (P1V1 - P2V2)/(gamma - 1). "
            "Isobaric (P = const): W = P*Delta_V = nR*Delta_T. "
            "Isochoric (V = const): W = 0. Q = Delta_U = n*Cv*Delta_T. "
            "Second Law of Thermodynamics: "
            "Kelvin-Planck: No engine can convert all heat into work (100% efficiency impossible). "
            "Clausius: Heat cannot spontaneously flow from cold to hot body. "
            "Carnot cycle: two isothermal + two adiabatic processes. "
            "Carnot efficiency: eta = 1 - T_cold/T_hot (maximum possible efficiency between two temperatures). "
            "Entropy: Delta_S = Q_rev/T. In irreversible processes, Delta_S_universe > 0."
        ),
    },
    {
        "id": "doc_012",
        "topic": "Transference of Heat",
        "text": (
            "Unit 12: Transference of Heat. "
            "Three modes of heat transfer: Conduction, Convection, Radiation. "
            "Conduction: heat transfer through a material without bulk motion. "
            "Fourier's law: Q/t = -K*A*(dT/dx). K = thermal conductivity (W/m.K). "
            "For a slab: Q/t = K*A*(T1 - T2)/L where L = thickness. "
            "Thermal resistance: R_th = L/(K*A). Series: R_total = R1 + R2. Parallel: 1/R_total = 1/R1 + 1/R2. "
            "Convection: heat transfer by bulk movement of fluid (natural vs forced convection). "
            "Radiation: heat transfer through electromagnetic waves (no medium needed). "
            "Black body: a perfect absorber and emitter of radiation. Absorptivity a = 1 for black body. "
            "Emissive power (E): energy radiated per unit area per unit time. "
            "Kirchhoff's Law: For any body at thermal equilibrium, emissive power / absorptive power = "
            "emissive power of a black body at same temperature. e/a = E_b. Good absorbers are good emitters. "
            "Wien's displacement law: lambda_max * T = b (Wien's constant b = 2.898 x 10^-3 m.K). "
            "As temperature increases, peak wavelength shifts to shorter wavelengths. "
            "Stefan-Boltzmann law: E = sigma * T^4 (for black body). "
            "sigma = 5.67 x 10^-8 W/(m^2.K^4). "
            "For a general body: E = epsilon * sigma * T^4 where epsilon = emissivity (0 < epsilon <= 1). "
            "Net radiation: P_net = epsilon*sigma*A*(T^4 - T_s^4) where T_s = surrounding temperature. "
            "Newton's law of cooling: dT/dt = -k*(T - T_s) (valid for small temperature differences). "
            "Solution: T(t) = T_s + (T_0 - T_s)*e^(-kt) (exponential decay toward T_s)."
        ),
    },
    {
        "id": "doc_013",
        "topic": "Electrostatics",
        "text": (
            "Unit 13: Electrostatics. "
            "Electric charge: fundamental property of matter. Unit: Coulomb (C). "
            "Quantization: charge exists in discrete multiples of e = 1.6 x 10^-19 C. q = n*e. "
            "Conservation: total charge in an isolated system remains constant. "
            "Coulomb's Law: F = k*q1*q2/r^2. k = 1/(4*pi*epsilon_0) = 9 x 10^9 N.m^2/C^2. "
            "epsilon_0 = 8.85 x 10^-12 C^2/(N.m^2). In a medium: F = k*q1*q2/(K*r^2), K = dielectric constant. "
            "Electric field: E = F/q_0 (force per unit positive test charge). Unit: N/C or V/m. "
            "Field due to point charge: E = k*q/r^2 (radially outward for +q). "
            "Electric dipole: two equal and opposite charges separated by distance 2a. Dipole moment p = q*2a (direction: -q to +q). "
            "Field on axial line of dipole: E = 2kp/r^3. Field on equatorial line: E = kp/r^3 (for r >> a). "
            "Torque on dipole in uniform field: tau = p x E, magnitude = pE*sin(theta). "
            "Electric flux: Phi = E . A = E*A*cos(theta). Unit: N.m^2/C (or V.m). "
            "Gauss's theorem: The total electric flux through a closed surface equals q_enclosed/epsilon_0. "
            "Phi = q_enc / epsilon_0. "
            "Applications of Gauss's law: "
            "Infinite line charge: E = lambda/(2*pi*epsilon_0*r). "
            "Infinite plane sheet: E = sigma/(2*epsilon_0). "
            "Spherical shell: E = kQ/r^2 (outside), E = 0 (inside). "
            "Electric potential: V = kq/r (scalar). Potential difference: V_A - V_B = W_AB/q. "
            "Relation: E = -dV/dr. "
            "Capacitance: C = Q/V. Unit: Farad (F). "
            "Parallel plate capacitor: C = epsilon_0*A/d. With dielectric: C = K*epsilon_0*A/d. "
            "Series: 1/C_eq = 1/C1 + 1/C2. Parallel: C_eq = C1 + C2. "
            "Energy stored: U = (1/2)*C*V^2 = (1/2)*Q*V = Q^2/(2C)."
        ),
    },
    {
        "id": "doc_014",
        "topic": "Current Electricity",
        "text": (
            "Unit 14: Current Electricity. "
            "Electric current: rate of flow of charge. I = dQ/dt. Unit: Ampere (A). "
            "Current density: J = I/A. Drift velocity: v_d = I/(n*A*e) = eE*tau/m. "
            "n = number density of electrons, tau = relaxation time. "
            "Sources of EMF: cells convert chemical/other energy to electrical energy. "
            "Primary cells: non-rechargeable (Leclanche, Daniel). "
            "Secondary cells: rechargeable (lead-acid, lithium-ion). "
            "EMF (electromotive force): potential difference across terminals when no current flows. "
            "Terminal voltage: V = EMF - I*r (r = internal resistance). "
            "Grouping of cells: "
            "Series: EMF_total = E1 + E2 + ..., r_total = r1 + r2 + ... "
            "Parallel: EMF_total = E (same), 1/r_total = 1/r1 + 1/r2 + ... "
            "Resistance: R = rho*L/A. rho = resistivity (unit: Ohm.m). "
            "Temperature dependence: R(T) = R_0*(1 + alpha*Delta_T). alpha = temperature coefficient. "
            "For metals: alpha > 0 (resistance increases with T). "
            "For semiconductors: alpha < 0 (resistance decreases with T). "
            "Ohm's Law: V = I*R (for ohmic conductors). "
            "Kirchhoff's Laws: "
            "Junction rule (KCL): sum of currents at a node = 0 (charge conservation). "
            "Loop rule (KVL): sum of potential differences around a closed loop = 0 (energy conservation). "
            "Series: R_eq = R1 + R2 + R3. Parallel: 1/R_eq = 1/R1 + 1/R2 + 1/R3. "
            "Wheatstone bridge: balanced when P/Q = R/S. No current through galvanometer. "
            "Potentiometer: compares EMFs. E1/E2 = L1/L2 (null-deflection method). "
            "Advantages: draws no current at balance, hence accurate."
        ),
    },
    {
        "id": "doc_015",
        "topic": "Thermal and Chemical Effects of Currents",
        "text": (
            "Unit 15: Thermal and Chemical Effects of Currents. "
            "Heating effect of current (Joule heating): H = I^2*R*t = V^2*t/R = V*I*t. "
            "Joule's law: heat produced is proportional to I^2, R, and time t. "
            "Electric power: P = V*I = I^2*R = V^2/R. Unit: Watt. "
            "Electrical energy: E = P*t. Unit: kWh (1 kWh = 3.6 x 10^6 J). "
            "Applications: electric heater, fuse, incandescent bulb. "
            "Thermoelectricity: Seebeck effect - when two different metals are joined to form a "
            "thermocouple and their junctions are maintained at different temperatures, an EMF is generated. "
            "Seebeck EMF: e = alpha*Delta_T + beta*(Delta_T)^2/2. "
            "Thermoelectric series: metals arranged such that current flows from earlier to later metal "
            "at the cold junction. "
            "Neutral temperature (T_n): temperature at which thermo-EMF is maximum. "
            "Inversion temperature (T_i): temperature at which thermo-EMF becomes zero. T_n = (T_c + T_i)/2. "
            "Peltier effect: heating or cooling at a junction when current passes through a thermocouple "
            "(reverse of Seebeck effect). "
            "Chemical effect of current (Electrolysis): "
            "Faraday's First Law: mass deposited m = Z*I*t where Z = electrochemical equivalent. "
            "Faraday's Second Law: m1/m2 = E1/E2 where E = chemical equivalent = atomic mass/valency. "
            "Faraday constant: F = 96485 C/mol. Z = E/F."
        ),
    },
    {
        "id": "doc_016",
        "topic": "Magnetic Effects of Currents",
        "text": (
            "Unit 16: Magnetic Effects of Currents. "
            "Oersted's experiment: a current-carrying conductor produces a magnetic field around it. "
            "Biot-Savart's Law: dB = (mu_0/(4*pi)) * (I*dl x r_hat)/r^2. "
            "mu_0 = 4*pi x 10^-7 T.m/A (permeability of free space). "
            "Magnetic field due to a long straight wire: B = mu_0*I/(2*pi*r). Direction by right-hand rule. "
            "Magnetic field at centre of a circular loop: B = mu_0*I/(2*R). "
            "For N turns: B = mu_0*N*I/(2*R). "
            "Magnetic field inside a solenoid: B = mu_0*n*I where n = N/L (turns per unit length). "
            "Lorentz force on a moving charge: F = q*(v x B). Magnitude: F = qvB*sin(theta). "
            "If v is parallel to B, F = 0. If v is perpendicular to B, particle moves in a circle. "
            "Radius of circular motion: r = mv/(qB). Time period: T = 2*pi*m/(qB) (independent of velocity). "
            "Force on a current-carrying conductor: F = I*(L x B). Magnitude: F = BIL*sin(theta). "
            "Torque on a current loop in magnetic field: tau = NIAB*sin(theta) = M x B. "
            "M = NIA = magnetic moment of the loop. "
            "Force between two parallel current-carrying wires: F/L = mu_0*I1*I2/(2*pi*d). "
            "Same direction currents attract, opposite direction currents repel. "
            "Definition of 1 Ampere: when F/L = 2 x 10^-7 N/m for d = 1m, I1 = I2 = 1A. "
            "Moving coil galvanometer: deflection phi = NIAB/(k) where k = torsional constant. "
            "Current sensitivity: phi/I = NAB/k. "
            "Conversion to ammeter: connect low resistance (shunt S) in parallel. S = I_g*G/(I - I_g). "
            "Conversion to voltmeter: connect high resistance R in series. R = (V/I_g) - G."
        ),
    },
    {
        "id": "doc_017",
        "topic": "Magnetostatics",
        "text": (
            "Unit 17: Magnetostatics. "
            "A bar magnet has a north and south pole. Magnetic field lines go from N to S outside "
            "the magnet and S to N inside. They form closed loops (no magnetic monopoles). "
            "Magnetic field intensity (H): H = B/mu_0 - M. Unit: A/m. "
            "Magnetic flux density (B): B = mu_0*(H + M) = mu_0*mu_r*H. "
            "Torque on a bar magnet in a uniform magnetic field: tau = M x B = MB*sin(theta). "
            "M = m*2l (magnetic moment, where m = pole strength, 2l = magnetic length). "
            "Earth's magnetic field: described by declination (angle between geographic and magnetic meridians) "
            "and inclination/dip (angle of total field with horizontal). "
            "Horizontal component: B_H = B*cos(delta). Vertical component: B_V = B*sin(delta). "
            "tan(delta) = B_V/B_H where delta = angle of dip. "
            "Classification of magnetic materials: "
            "Diamagnetic: weakly repelled by magnets. chi < 0, mu_r < 1. Examples: bismuth, copper, water. "
            "Paramagnetic: weakly attracted by magnets. chi > 0 (small), mu_r > 1 (slightly). "
            "Examples: aluminium, platinum, oxygen. chi varies as 1/T (Curie's law). "
            "Ferromagnetic: strongly attracted by magnets. chi >> 1, mu_r >> 1. "
            "Examples: iron, cobalt, nickel. Show hysteresis, domain structure, Curie temperature. "
            "Magnetic susceptibility: chi = M/H (ratio of magnetization to field intensity). "
            "Relative permeability: mu_r = 1 + chi. "
            "Magnetic induction: B = mu_0*(1 + chi)*H = mu_0*mu_r*H. "
            "Hysteresis: B-H curve for ferromagnetic materials showing retentivity and coercivity."
        ),
    },
    {
        "id": "doc_018",
        "topic": "Electromagnetic Induction and Alternating Currents",
        "text": (
            "Unit 18: Electromagnetic Induction and Alternating Currents. "
            "Faraday's Law of Electromagnetic Induction: induced EMF = -d(Phi_B)/dt. "
            "Phi_B = B*A*cos(theta) = magnetic flux. "
            "Lenz's Law: induced current opposes the change that produces it (conservation of energy). "
            "Motional EMF: EMF = B*l*v (for a conductor moving in a magnetic field). "
            "Self-inductance: EMF = -L*(dI/dt). L = N*Phi/I. Unit: Henry (H). "
            "Inductance of a solenoid: L = mu_0*n^2*A*l = mu_0*N^2*A/l. "
            "Energy stored in inductor: U = (1/2)*L*I^2. "
            "Mutual inductance: EMF_2 = -M*(dI_1/dt). M = k*sqrt(L1*L2) where k = coupling coefficient. "
            "Alternating current: I = I_0*sin(omega*t). V = V_0*sin(omega*t). "
            "RMS values: I_rms = I_0/sqrt(2). V_rms = V_0/sqrt(2). "
            "Purely resistive circuit: V and I in phase. "
            "Purely inductive circuit: V leads I by 90 degrees. X_L = omega*L (inductive reactance). "
            "Purely capacitive circuit: I leads V by 90 degrees. X_C = 1/(omega*C) (capacitive reactance). "
            "LCR series circuit: Impedance Z = sqrt(R^2 + (X_L - X_C)^2). "
            "Phase angle: tan(phi) = (X_L - X_C)/R. "
            "Power: P = V_rms*I_rms*cos(phi). cos(phi) = R/Z (power factor). "
            "Resonance: when X_L = X_C, i.e. omega_0 = 1/sqrt(LC). "
            "At resonance: Z = R (minimum), I = maximum, power factor = 1. "
            "Quality factor: Q = omega_0*L/R = 1/(omega_0*C*R) = (1/R)*sqrt(L/C). "
            "Transformer: V_s/V_p = N_s/N_p = I_p/I_s. Step-up: N_s > N_p. Step-down: N_s < N_p. "
            "AC generator: EMF = NBA*omega*sin(omega*t). Peak EMF = NBA*omega."
        ),
    },
    {
        "id": "doc_019",
        "topic": "Ray Optics",
        "text": (
            "Unit 19: Ray Optics. "
            "Reflection: angle of incidence = angle of reflection. "
            "Mirror formula: 1/v + 1/u = 1/f. Magnification: m = -v/u. "
            "Concave mirror: f < 0 (converging). Convex mirror: f > 0 (diverging). "
            "Sign convention: distances measured from pole, along principal axis, in direction of incident light are positive. "
            "Refraction: Snell's law: n1*sin(i) = n2*sin(i_r). n = c/v (refractive index). "
            "Total internal reflection: occurs when light goes from denser to rarer medium "
            "and angle of incidence > critical angle. sin(i_c) = n2/n1 (n1 > n2). "
            "Optical fibre: uses total internal reflection to guide light through thin glass fibres. "
            "Principles: light enters within acceptance cone, reflects internally, minimal loss. "
            "Deviation by a prism: delta = (i + e) - A where A = angle of prism. "
            "At minimum deviation: i = e and r1 = r2 = A/2. "
            "n = sin((A + delta_min)/2) / sin(A/2). "
            "Dispersion: splitting of white light into constituent colours. "
            "Cauchy's equation: n = A + B/lambda^2 (n varies with wavelength). "
            "Angular dispersion = delta_V - delta_R. "
            "Lens formula: 1/v - 1/u = 1/f. Magnification: m = v/u. "
            "Lensmaker's equation: 1/f = (n - 1)*(1/R1 - 1/R2). "
            "Power of lens: P = 1/f (in dioptres when f is in metres). "
            "Combination: P = P1 + P2. 1/f = 1/f1 + 1/f2. "
            "Resolving power: ability to distinguish two closely spaced objects. "
            "Microscope: magnification = (1 + D/f_e) * (-v_o/u_o) for compound microscope. "
            "Telescope: magnification = f_o/f_e (normal adjustment)."
        ),
    },
    {
        "id": "doc_020",
        "topic": "Wave Optics",
        "text": (
            "Unit 20: Wave Optics. "
            "Wave nature of light: light is a transverse electromagnetic wave. "
            "Huygens' principle: every point on a wavefront acts as a source of secondary wavelets. "
            "The new wavefront is the envelope of these wavelets. "
            "Interference: superposition of two coherent waves. "
            "Constructive: path difference = n*lambda (bright fringe). "
            "Destructive: path difference = (2n+1)*lambda/2 (dark fringe). "
            "Young's Double Slit Experiment (YDSE): "
            "Fringe width: beta = lambda*D/d where D = distance to screen, d = slit separation. "
            "Position of nth bright fringe: y_n = n*lambda*D/d. "
            "Position of nth dark fringe: y_n = (2n-1)*lambda*D/(2d). "
            "Intensity: I = 4*I_0*cos^2(phi/2) where phi = 2*pi*Delta_x/lambda. "
            "Conditions for sustained interference: sources must be coherent (same frequency, constant phase difference), "
            "monochromatic, and of comparable amplitudes. "
            "Diffraction: bending of light around obstacles or through narrow openings. "
            "Single slit diffraction: condition for minima: a*sin(theta) = n*lambda (n = 1, 2, 3...). "
            "Central maximum is twice as wide as other maxima. "
            "Angular width of central maximum: 2*lambda/a. "
            "Polarization: vibration of light restricted to one plane. "
            "Malus' Law: I = I_0*cos^2(theta) (intensity after polarizer at angle theta). "
            "Brewster's Law: tan(i_p) = n (reflected light is completely polarized at Brewster's angle). "
            "Unpolarized light through a polarizer reduces intensity by half: I = I_0/2."
        ),
    },
    {
        "id": "doc_021",
        "topic": "Electromagnetic Waves",
        "text": (
            "Unit 21: Electromagnetic Waves. "
            "Maxwell's displacement current: I_d = epsilon_0*(d(Phi_E)/dt). "
            "This completes Ampere's law for time-varying fields. "
            "Electromagnetic waves: oscillating electric and magnetic fields perpendicular to each other "
            "and to the direction of propagation. They are transverse waves. "
            "Speed: c = 1/sqrt(mu_0*epsilon_0) = 3 x 10^8 m/s. E_0/B_0 = c. "
            "EM waves do not require a medium for propagation. "
            "Properties: carry energy, momentum (p = U/c), exert radiation pressure (P = I/c). "
            "Electromagnetic spectrum (increasing frequency, decreasing wavelength): "
            "Radio waves (> 0.1 m): communication, broadcasting. "
            "Microwaves (0.1 m to 1 mm): radar, cooking, satellite communication. "
            "Infrared (1 mm to 700 nm): thermal imaging, remote controls, night vision. "
            "Visible light (700 nm to 400 nm): VIBGYOR - human vision. "
            "Ultraviolet (400 nm to 10 nm): sterilization, fluorescence, vitamin D synthesis. "
            "X-rays (10 nm to 0.01 nm): medical imaging, crystallography. "
            "Gamma rays (< 0.01 nm): nuclear reactions, cancer treatment, sterilization. "
            "Propagation in atmosphere: "
            "Ground/surface wave: follows Earth's curvature (AM radio). "
            "Sky wave: reflected by ionosphere (shortwave radio, long distance). "
            "Space/line-of-sight wave: travels straight (FM, TV, satellite). "
            "Energy density: u = (1/2)*epsilon_0*E^2 + (1/2)*B^2/mu_0. "
            "Intensity: I = (1/2)*c*epsilon_0*E_0^2."
        ),
    },
    {
        "id": "doc_022",
        "topic": "Electron and Photons",
        "text": (
            "Unit 22: Electron and Photons. "
            "Charge on electron: e = 1.6 x 10^-19 C (Millikan's oil drop experiment). "
            "Specific charge (e/m) of electron: determined by J.J. Thomson's experiment using "
            "crossed electric and magnetic fields. e/m = 1.76 x 10^11 C/kg. "
            "When E and B balanced: v = E/B. Then: e/m = v/(B*r) = E/(B^2*r). "
            "Photoelectric effect: emission of electrons when light falls on a metal surface. "
            "Observations: (1) instantaneous emission (no time lag), "
            "(2) KE of electrons depends on frequency, not intensity, "
            "(3) exists a threshold frequency below which no emission occurs, "
            "(4) number of electrons proportional to intensity. "
            "Einstein's photoelectric equation: KE_max = h*f - phi = h*(f - f_0). "
            "h = Planck's constant = 6.626 x 10^-34 J.s. "
            "phi = work function = h*f_0 (minimum energy to eject electron). f_0 = threshold frequency. "
            "Stopping potential: eV_0 = KE_max = h*f - phi. "
            "Energy of photon: E = h*f = hc/lambda. "
            "Momentum of photon: p = h/lambda = E/c. "
            "Wave-particle duality: light shows both wave nature (interference, diffraction) "
            "and particle nature (photoelectric effect, Compton effect). "
            "Photon has zero rest mass, always moves at speed c."
        ),
    },
    {
        "id": "doc_023",
        "topic": "Atoms, Molecules and Nuclei",
        "text": (
            "Unit 23: Atoms, Molecules and Nuclei. "
            "Rutherford's alpha particle scattering: Most alpha particles pass through, few deflect. "
            "Conclusion: atom has a tiny, dense, positive nucleus with electrons orbiting around it. "
            "Impact parameter: b = (kZe^2/(KE)) * cot(theta/2). "
            "Distance of closest approach: r_0 = kZe^2/(KE_alpha). "
            "Atomic mass unit: 1 u = 1.66 x 10^-27 kg = 931.5 MeV/c^2. "
            "Size of nucleus: R = R_0 * A^(1/3) where R_0 = 1.2 fm, A = mass number. "
            "Nuclear density is constant: rho = 3m_p/(4*pi*R_0^3) = approx 2.3 x 10^17 kg/m^3. "
            "Radioactivity: spontaneous disintegration of unstable nuclei. "
            "Alpha decay: Z reduces by 2, A reduces by 4. X -> Y + He-4. "
            "Beta-minus decay: neutron -> proton + electron + antineutrino. Z increases by 1. "
            "Beta-plus decay: proton -> neutron + positron + neutrino. Z decreases by 1. "
            "Gamma decay: excited nucleus emits a photon, no change in Z or A. "
            "Radioactive decay law: N = N_0 * e^(-lambda*t). Activity: A = lambda*N = A_0*e^(-lambda*t). "
            "Half-life: T_1/2 = 0.693/lambda. Mean life: tau = 1/lambda = T_1/2/0.693. "
            "Binding energy: energy needed to separate a nucleus into individual nucleons. "
            "BE = [Z*m_p + (A-Z)*m_n - M_nucleus] * c^2. "
            "Mass-energy equivalence: E = mc^2 (Einstein). "
            "Binding energy per nucleon: maximum near A = 56 (Fe), indicating maximum stability. "
            "Nuclear fission: heavy nucleus splits into lighter nuclei (U-235 + n -> Ba + Kr + 3n + energy). "
            "Chain reaction, critical mass, nuclear reactor. "
            "Nuclear fusion: light nuclei combine to form heavier nucleus (4H -> He + 2e+ + 2nu + 26.7 MeV). "
            "Powers the Sun. Requires extremely high temperature (thermonuclear)."
        ),
    },
    {
        "id": "doc_024",
        "topic": "Solids and Semiconductor Devices",
        "text": (
            "Unit 24: Solids and Semiconductor Devices. "
            "Energy bands in solids: arise from splitting of atomic energy levels when atoms form a crystal. "
            "Valence band: highest occupied energy band. Conduction band: lowest unoccupied band. "
            "Band gap (E_g): energy difference between conduction and valence band. "
            "Conductors: valence and conduction bands overlap. E_g = 0. High conductivity. "
            "Insulators: large band gap (E_g > 3 eV). Very low conductivity. Example: diamond (E_g = 5.5 eV). "
            "Semiconductors: small band gap (E_g approx 1 eV). Si: E_g = 1.1 eV, Ge: E_g = 0.67 eV. "
            "Intrinsic semiconductor: pure, n_e = n_h = n_i. Conductivity increases with temperature. "
            "n-type: doped with pentavalent (P, As, Sb). Electrons are majority carriers. "
            "p-type: doped with trivalent (B, Al, Ga). Holes are majority carriers. "
            "p-n junction: formed when p-type and n-type semiconductors are joined. "
            "Depletion region: thin layer at junction devoid of free charges. Built-in potential barrier forms. "
            "Forward bias: p connected to +ve, n to -ve. Depletion region narrows, current flows easily. "
            "Reverse bias: p connected to -ve, n to +ve. Depletion region widens, very small leakage current. "
            "Diode as rectifier: allows current in one direction only. "
            "Half-wave rectifier: uses one diode, output frequency = input frequency. "
            "Full-wave rectifier: uses two diodes (or bridge of four), output frequency = 2 * input frequency. "
            "Transistor: three-layer device (npn or pnp). Regions: emitter, base, collector. "
            "Transistor action: thin, lightly doped base allows most carriers to cross from emitter to collector. "
            "I_E = I_B + I_C. Current gain: beta = I_C/I_B (typically 20-200). "
            "Transistor as amplifier: small change in base current causes large change in collector current. "
            "Voltage gain: A_v = beta * R_C/R_B. Common-emitter configuration most widely used."
        ),
    },
    {
        "id": "doc_025",
        "topic": "Damped and Forced Oscillations",
        "text": (
            "Unit 25: Damped and Forced Oscillations. "
            "Damped harmonic oscillation: oscillation with a resistive (damping) force proportional to velocity. "
            "Equation: m*(d^2x/dt^2) + b*(dx/dt) + kx = 0, where b = damping coefficient. "
            "General solution: x(t) = A*e^(-gamma*t)*cos(omega_d*t + phi) where gamma = b/(2m). "
            "Damped frequency: omega_d = sqrt(omega_0^2 - gamma^2) where omega_0 = sqrt(k/m). "
            "Three cases based on damping: "
            "(1) Underdamped (b^2 < 4mk or gamma < omega_0): oscillatory with exponentially decaying amplitude. "
            "System oscillates but amplitude decreases over time. Most common in real physical systems. "
            "(2) Overdamped (b^2 > 4mk or gamma > omega_0): no oscillation, system returns to equilibrium "
            "very slowly. x(t) = A1*e^(-alpha_1*t) + A2*e^(-alpha_2*t). "
            "(3) Critically damped (b^2 = 4mk or gamma = omega_0): fastest return to equilibrium without oscillation. "
            "x(t) = (A + Bt)*e^(-gamma*t). Used in door closers, car suspensions, galvanometers. "
            "Energy decay: E(t) = E_0*e^(-2*gamma*t) = E_0*e^(-t/tau_E). Energy decays exponentially. "
            "Relaxation time (tau): time for amplitude to fall to 1/e of initial value. tau = m/b = 1/(2*gamma). "
            "Quality factor: Q = 2*pi * (energy stored)/(energy lost per cycle) = omega_0/(2*gamma). "
            "High Q means low damping, sharp resonance. Low Q means heavy damping, broad resonance. "
            "Forced oscillation: external periodic force F = F_0*cos(omega*t) applied to a damped system. "
            "Equation: m*(d^2x/dt^2) + b*(dx/dt) + kx = F_0*cos(omega*t). "
            "Steady-state solution: x(t) = A*cos(omega*t - delta). "
            "Amplitude: A = F_0 / sqrt((k - m*omega^2)^2 + (b*omega)^2). "
            "Resonance: maximum amplitude when driving frequency approaches natural frequency. "
            "At resonance (omega = omega_0): amplitude is maximum = F_0/(b*omega_0). "
            "Applications: tuning circuits, RLC circuits, musical instruments, bridges."
        ),
    },
    {
        "id": "doc_026",
        "topic": "Waves and Interference",
        "text": (
            "Unit 26: Waves and Interference. "
            "Wave equation: d^2y/dx^2 = (1/v^2) * d^2y/dt^2 (one-dimensional wave equation). "
            "General solution: y(x,t) = f(x - vt) + g(x + vt). "
            "Superposition of waves: when two or more waves overlap, the resultant displacement "
            "is the vector sum of individual displacements. "
            "Interference of light: redistribution of energy due to superposition of coherent waves. "
            "Constructive interference: waves in phase, path difference = n*lambda: I_max = (A1 + A2)^2. "
            "Destructive interference: waves out of phase, path difference = (2n+1)*lambda/2: I_min = (A1 - A2)^2. "
            "Resultant intensity: I = I1 + I2 + 2*sqrt(I1*I2)*cos(delta) where delta = phase difference. "
            "Types of interference: "
            "(1) Division of wavefront: a wavefront is divided by slits, edges, or mirrors. "
            "Examples: Young's double slit, Lloyd's mirror, Fresnel biprism. "
            "(2) Division of amplitude: the amplitude of an incoming wave is divided by partial reflection. "
            "Examples: thin film interference, Newton's rings, Michelson interferometer. "
            "Coherence: two sources are coherent if they have the same frequency and a constant phase relationship. "
            "Temporal coherence: related to monochromaticity of the source. Coherence length = c * coherence time. "
            "Spatial coherence: related to the size of the source."
        ),
    },
    {
        "id": "doc_027",
        "topic": "Interference in Thin Films",
        "text": (
            "Unit 27: Interference in Thin Films. "
            "When light falls on a thin transparent film, reflections from the top and bottom surfaces "
            "interfere with each other. A phase change of pi (half wavelength) occurs on reflection from "
            "a denser medium. "
            "For reflected light: "
            "Constructive: 2*mu*t*cos(r) = (2n+1)*lambda/2 (bright, accounting for phase change). "
            "Destructive: 2*mu*t*cos(r) = n*lambda (dark). "
            "For transmitted light: conditions are reversed. "
            "mu = refractive index of film, t = thickness, r = angle of refraction. "
            "Wedge-shaped thin film: film of variable thickness. "
            "Fringe width: beta = lambda/(2*mu*theta) where theta = wedge angle. "
            "Fringes are localized, straight, parallel, and equidistant. "
            "Newton's rings: circular interference fringes formed between a planoconvex lens and a flat glass plate. "
            "An air film of varying thickness produces concentric bright and dark rings. "
            "Dark ring radius: r_n = sqrt(n*lambda*R) where R = radius of curvature of lens. "
            "Bright ring radius: r_n = sqrt((2n-1)*lambda*R/2). "
            "The central spot is dark in reflected light (due to phase change on reflection). "
            "Applications: measuring wavelength, refractive index of liquids, testing optical flatness. "
            "Diameter of nth dark ring: D_n = 2*sqrt(n*lambda*R). D_n^2 = 4*n*lambda*R. "
            "For liquid of refractive index mu between lens and plate: D_n^2 = 4*n*lambda*R/mu. "
            "Michelson interferometer: uses division of amplitude. Two mirrors M1 and M2 with a beam splitter. "
            "Path difference = 2*d*cos(theta) where d = mirror displacement. "
            "For bright fringes: 2*d*cos(theta) = n*lambda. "
            "Applications: precise measurement of wavelength, thickness of thin films, "
            "refractive index of gases, measuring the standard metre."
        ),
    },
    {
        "id": "doc_028",
        "topic": "Diffraction",
        "text": (
            "Unit 28: Diffraction. "
            "Diffraction: bending of waves around obstacles or through narrow apertures when their size "
            "is comparable to the wavelength. "
            "Applications: X-ray crystallography, holography, spectrometers, CD/DVD data reading. "
            "Types of diffraction: "
            "(1) Fresnel diffraction: source or screen (or both) at finite distance. Wavefronts are spherical/cylindrical. "
            "(2) Fraunhofer diffraction: source and screen at effectively infinite distance (parallel rays). "
            "Achieved using lenses. Mathematically simpler. "
            "Fraunhofer diffraction by a single slit of width a: "
            "Condition for minima: a*sin(theta) = n*lambda (n = 1, 2, 3...). "
            "Condition for secondary maxima: a*sin(theta) = (2n+1)*lambda/2 (n = 1, 2, 3...). "
            "Central maximum angular width: 2*lambda/a. Linear width on screen: 2*f*lambda/a. "
            "Intensity distribution: I = I_0 * (sin(beta)/beta)^2 where beta = (pi*a*sin(theta))/lambda. "
            "Central maximum is brightest, secondary maxima rapidly decrease in intensity. "
            "Plane diffraction grating: N parallel equidistant slits. "
            "Grating element: d = a + b (slit width + opaque space). "
            "Principal maxima condition: d*sin(theta) = n*lambda (n = 0, 1, 2...). "
            "Condition for minima: d*sin(theta) = m*lambda/N (m is not a multiple of N). "
            "Maximum order observable: n_max = d/lambda (since sin(theta) <= 1). "
            "Absent spectra: if d/a = integer, certain orders are missing. "
            "e.g., if a = b (d = 2a), orders n = 2, 4, 6... are absent. "
            "Dispersive power: d(theta)/d(lambda) = n/(d*cos(theta)). "
            "Higher order and smaller grating element give greater dispersion. "
            "Resolving power of grating: R = lambda/d(lambda) = n*N (n = order, N = total slits)."
        ),
    },
    {
        "id": "doc_029",
        "topic": "Quantum Mechanics",
        "text": (
            "Unit 29: Quantum Mechanics. "
            "Dual nature of radiation: light exhibits both wave (interference, diffraction) and "
            "particle (photoelectric, Compton) behaviour. "
            "de Broglie hypothesis: every moving particle has an associated wave. "
            "de Broglie wavelength: lambda = h/p = h/(mv). "
            "For electron accelerated through V volts: lambda = 1.226/sqrt(V) nm. "
            "Phase velocity: v_p = omega/k = E/p. Group velocity: v_g = d(omega)/dk = dE/dp = v (particle velocity). "
            "Relation: v_p * v_g = c^2 (for relativistic particles). v_p > c but v_g < c always. "
            "Heisenberg's Uncertainty Principle: "
            "Delta_x * Delta_p >= h_bar/2 (position-momentum). "
            "Delta_E * Delta_t >= h_bar/2 (energy-time). "
            "h_bar = h/(2*pi) = 1.055 x 10^-34 J.s. "
            "Applications: explains why electrons cannot exist inside nucleus, zero-point energy of oscillators, "
            "natural line width of spectral lines. "
            "Wave function Psi(x,t): describes quantum state of a particle. |Psi|^2 gives probability density. "
            "Normalization: integral of |Psi|^2 dx from -inf to +inf = 1. "
            "Operators: physical observables are represented by operators. "
            "Position operator: x_hat = x. Momentum operator: p_hat = -i*h_bar*(d/dx). "
            "Energy operator (Hamiltonian): H_hat = -(h_bar^2/(2m))*(d^2/dx^2) + V(x). "
            "Schrodinger's time-dependent equation: i*h_bar*(d(Psi)/dt) = H_hat * Psi. "
            "Time-independent equation: H_hat * psi = E * psi, i.e. -(h_bar^2/(2m))*(d^2(psi)/dx^2) + V*psi = E*psi. "
            "Postulates of QM: (1) state described by Psi, (2) observables are operators, "
            "(3) measurement yields eigenvalues, (4) Psi evolves via Schrodinger equation. "
            "Particle in a 1D box (infinite potential well, width L): "
            "Energy levels: E_n = n^2*pi^2*h_bar^2/(2mL^2) = n^2*h^2/(8mL^2), n = 1, 2, 3... "
            "Wave functions: psi_n = sqrt(2/L)*sin(n*pi*x/L). "
            "Zero-point energy: E_1 = h^2/(8mL^2) (particle can never have zero energy). "
            "Applications: quantum dots, conjugated molecules, nanoscale devices. "
            "Quantum tunnelling: a particle can penetrate a potential barrier even if E < V. "
            "Transmission coefficient: T = e^(-2*kappa*L) where kappa = sqrt(2m(V-E))/h_bar. "
            "Applications: tunnel diode, scanning tunnelling microscope (STM), alpha decay, nuclear fusion in stars."
        ),
    },
    {
        "id": "doc_030",
        "topic": "Electromagnetic Theory",
        "text": (
            "Unit 30: Electromagnetic Theory. "
            "Vector calculus fundamentals: "
            "Gradient: grad(f) = (df/dx)i + (df/dy)j + (df/dz)k. Points in direction of maximum increase of f. "
            "Divergence: div(A) = dAx/dx + dAy/dy + dAz/dz. Scalar. Measures source/sink strength. "
            "Curl: curl(A) = (dAz/dy - dAy/dz)i + (dAx/dz - dAz/dx)j + (dAy/dx - dAx/dy)k. Measures rotation. "
            "Line integral: integral of A . dl along a curve. "
            "Surface integral: integral of A . dS over a surface (flux). "
            "Volume integral: integral of f dV over a volume. "
            "Gauss divergence theorem: integral of A . dS (closed surface) = integral of div(A) dV (volume). "
            "Converts surface integral to volume integral. "
            "Stokes' theorem: integral of A . dl (closed loop) = integral of curl(A) . dS (surface). "
            "Converts line integral to surface integral. "
            "Maxwell's equations (differential form): "
            "(1) div(E) = rho/epsilon_0 (Gauss's law for electricity). "
            "(2) div(B) = 0 (Gauss's law for magnetism - no magnetic monopoles). "
            "(3) curl(E) = -dB/dt (Faraday's law). "
            "(4) curl(B) = mu_0*J + mu_0*epsilon_0*dE/dt (Ampere-Maxwell law). "
            "Maxwell's equations (integral form): "
            "(1) integral E . dA = Q_enc/epsilon_0. "
            "(2) integral B . dA = 0. "
            "(3) integral E . dl = -d(Phi_B)/dt. "
            "(4) integral B . dl = mu_0*I_enc + mu_0*epsilon_0*d(Phi_E)/dt. "
            "EM wave equations derived from Maxwell's equations (in free space, J=0, rho=0): "
            "del^2(E) = mu_0*epsilon_0 * d^2(E)/dt^2. "
            "del^2(B) = mu_0*epsilon_0 * d^2(B)/dt^2. "
            "Wave speed: v = 1/sqrt(mu_0*epsilon_0) = c = 3 x 10^8 m/s. "
            "Transverse nature: E, B, and direction of propagation are mutually perpendicular. "
            "E x B gives the direction of propagation (Poynting vector: S = (1/mu_0) * E x B)."
        ),
    },
    {
        "id": "doc_031",
        "topic": "Laser and Fiber Optics - LASER",
        "text": (
            "Unit 31: LASER (Light Amplification by Stimulated Emission of Radiation). "
            "Properties of laser light: (1) monochromatic (single wavelength), "
            "(2) coherent (constant phase relationship), (3) highly directional (low divergence), "
            "(4) high intensity (concentrated energy). "
            "Applications: communication, surgery, welding, barcode reading, holography, spectroscopy, "
            "CD/DVD, laser printing, military, nuclear fusion research. "
            "Absorption: atom in ground state absorbs photon and goes to excited state. "
            "Rate of absorption: R_abs = B_12 * N_1 * u(f) where B_12 = Einstein B coefficient, "
            "N_1 = population of lower level, u(f) = energy density of radiation. "
            "Spontaneous emission: atom in excited state spontaneously drops to lower state and emits a photon. "
            "Rate: R_sp = A_21 * N_2. The emitted photons are random in direction and phase (incoherent). "
            "Stimulated emission: an incoming photon stimulates the excited atom to emit an identical photon. "
            "The emitted photon has the same frequency, phase, direction, and polarization as the incoming photon. "
            "Rate: R_st = B_21 * N_2 * u(f). This is the basis of laser action. "
            "Meta-stable state: an excited state with a relatively long lifetime (milliseconds vs nanoseconds). "
            "Necessary for achieving population inversion. "
            "Population inversion: N_2 > N_1 (more atoms in excited state than ground state). "
            "Cannot be achieved in thermal equilibrium (violates Boltzmann distribution). "
            "Pumping: the process of achieving population inversion. Methods: optical, electrical, chemical. "
            "Three-level laser: ground state, meta-stable state, pump level. "
            "Atoms are pumped to highest level, rapidly decay to meta-stable state, population inversion "
            "between meta-stable and ground state. Example: Ruby Laser. "
            "Four-level laser: more efficient because lower laser level is not the ground state, "
            "so population inversion is easier to achieve. Examples: He-Ne laser, Nd:YAG laser. "
            "Ruby Laser: solid-state, three-level, uses ruby crystal (Al2O3 doped with Cr3+ ions). "
            "Pumped by xenon flash lamp. Wavelength: 694.3 nm (red). Pulsed output. "
            "Components: ruby rod, flash lamp, optical resonator (fully and partially reflecting mirrors)."
        ),
    },
    {
        "id": "doc_032",
        "topic": "Optical Fiber",
        "text": (
            "Unit 32: Optical Fiber. "
            "Principle: light propagation through total internal reflection inside a thin glass/plastic fibre. "
            "Construction: three layers - (1) Core: innermost, high refractive index (n1), carries light. "
            "(2) Cladding: surrounds core, lower refractive index (n2 < n1), ensures total internal reflection. "
            "(3) Buffer/jacket: outer protective coating. "
            "Types of optical fibres: "
            "(1) Step-index single mode: very thin core (8-10 micrometres), only one mode propagates. "
            "Lowest dispersion, highest bandwidth, used for long-distance communication. "
            "(2) Step-index multimode: larger core (50-200 micrometres), many modes propagate. "
            "Higher dispersion (modal dispersion), easier to couple light, shorter distances. "
            "(3) Graded-index multimode: refractive index decreases gradually from centre to edge. "
            "Reduces modal dispersion as different modes travel different paths but at different speeds, "
            "arriving at nearly the same time. "
            "Acceptance angle (theta_a): maximum angle at which light can enter the fibre and still "
            "undergo total internal reflection inside. "
            "sin(theta_a) = sqrt(n1^2 - n2^2) / n0 where n0 = refractive index of surrounding medium. "
            "Numerical aperture (NA): sin(theta_a) = NA = sqrt(n1^2 - n2^2). "
            "NA determines the light-gathering ability of the fibre. "
            "For small index difference: NA = n1*sqrt(2*Delta) where Delta = (n1 - n2)/n1 (fractional index difference). "
            "V-number (normalized frequency): V = (pi*d/lambda)*NA. "
            "Single mode condition: V < 2.405. Number of modes approximately V^2/2 for large V. "
            "Losses in optical fibres: absorption (material impurities), scattering (Rayleigh scattering), "
            "bending losses (micro and macro bending), dispersion (modal, material, waveguide). "
            "Applications: telecommunications (high bandwidth, low loss, immune to EMI), "
            "medical endoscopy, sensors (temperature, pressure, strain), military, "
            "internet backbone, cable TV, local area networks (LAN)."
        ),
    },
]


# Shared Embedder Instance
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ChromaDB Collection Setup
_collection = None

def get_collection():
    global _collection
    if _collection is not None:
        return _collection

    embedder = get_embedder()
    client = chromadb.Client()

    # Delete existing collection if present
    try:
        client.delete_collection("study_buddy_kb")
    except Exception:
        pass

    collection = client.create_collection("study_buddy_kb")

    texts = [d["text"] for d in DOCUMENTS]
    ids = [d["id"] for d in DOCUMENTS]
    embeddings = embedder.encode(texts).tolist()
    metadatas = [{"topic": d["topic"]} for d in DOCUMENTS]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    _collection = collection
    return _collection


if __name__ == "__main__":
    collection = get_collection()
    print(f"Knowledge base ready: {collection.count()} documents")
    for d in DOCUMENTS:
        print(f"  - {d['topic']}")
