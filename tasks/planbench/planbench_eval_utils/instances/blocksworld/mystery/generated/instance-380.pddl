(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects c l j e d g a f k i b h)
(:init 
(harmony)
(planet c)
(planet l)
(planet j)
(planet e)
(planet d)
(planet g)
(planet a)
(planet f)
(planet k)
(planet i)
(planet b)
(planet h)
(province c)
(province l)
(province j)
(province e)
(province d)
(province g)
(province a)
(province f)
(province k)
(province i)
(province b)
(province h)
)
(:goal
(and
(craves c l)
(craves l j)
(craves j e)
(craves e d)
(craves d g)
(craves g a)
(craves a f)
(craves f k)
(craves k i)
(craves i b)
(craves b h)
)))