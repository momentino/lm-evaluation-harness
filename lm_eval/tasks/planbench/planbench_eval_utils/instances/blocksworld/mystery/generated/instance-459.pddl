(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects b d a k g i h l f)
(:init 
(harmony)
(planet b)
(planet d)
(planet a)
(planet k)
(planet g)
(planet i)
(planet h)
(planet l)
(planet f)
(province b)
(province d)
(province a)
(province k)
(province g)
(province i)
(province h)
(province l)
(province f)
)
(:goal
(and
(craves b d)
(craves d a)
(craves a k)
(craves k g)
(craves g i)
(craves i h)
(craves h l)
(craves l f)
)))