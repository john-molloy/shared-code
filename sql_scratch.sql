


select coalesce(null, null, 123, 456, null)             -- Returns 123
select coalesce(123, 456)                               -- Returns 123
select coalesce(123)                                    -- ERROR: "Incorrect syntax near ')'."
select coalesce(null, null, null)                       -- ERROR: "At least one of the arguments to COALESCE must be an expression that is not the NULL constant."
select coalesce(null, null, cast(null as integer))      -- Returns NULL

declare @t as varchar(100)

declare @i as integer
set @i = coalesce(123, 456)
select @i
set @i = coalesce(null, null, cast(null as integer))
select @i
set @i = coalesce(null, null, cast(null as varchar))    -- This works
select @i
set @i = coalesce(null, null, 123, 'abc')
select @i


