(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects l g c e k)
(:init 
(harmony)
(planet l)
(planet g)
(planet c)
(planet e)
(planet k)
(province l)
(province g)
(province c)
(province e)
(province k)
)
(:goal
(and
(craves l g)
(craves g c)
(craves c e)
(craves e k)
)))