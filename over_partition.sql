
drop table #SupporterTable
drop table #IncomeTable

create table #SupporterTable
(
    supporter_urn   int             not null,
    [name]          varchar(100)    not null,
    gender          varchar(4)      not null,
    age_band        varchar(16)     not null
)

create table #IncomeTable
(
    supporter_urn   int             not null,
    amount          float           not null,
    banking_date    date            not null,
    dept            int             not null,
    payment_method  varchar(32)     not null
)

insert into #SupporterTable values( 1, 'Aaron',    'm', '18 to 25')
insert into #SupporterTable values( 2, 'Alana',    'f', '18 to 25')
insert into #SupporterTable values( 3, 'Bob',      'm', '26 to 35')
insert into #SupporterTable values( 4, 'Barbara',  'f', '26 to 35')
--insert into #SupporterTable values( 5, 'Charles',  'm', '36 to 45')
--insert into #SupporterTable values( 6, 'Caroline', 'f', '36 to 45')
--insert into #SupporterTable values( 7, 'David',    'm', '46 to 55')
--insert into #SupporterTable values( 8, 'Deborah',  'f', '46 to 55')
--insert into #SupporterTable values( 9, 'Eric',     'm', '56 to 65')
--insert into #SupporterTable values(10, 'Erica',    'f', '56 to 65')

insert into #IncomeTable values (1, 10.0, '2021-01-01', 1, 'COG')
insert into #IncomeTable values (1, 20.0, '2021-02-01', 1, 'COG')
insert into #IncomeTable values (2, 30.0, '2021-01-01', 1, 'COG')
insert into #IncomeTable values (2, 40.0, '2021-02-01', 1, 'Cash')
insert into #IncomeTable values (2, 50.0, '2021-03-01', 1, 'Cash')
insert into #IncomeTable values (3, 60.0, '2021-01-01', 1, 'Cash')
insert into #IncomeTable values (3, 70.0, '2021-02-01', 1, 'COG')
insert into #IncomeTable values (3, 80.0, '2021-03-01', 1, 'Cash')


select * from #SupporterTable
select * from #IncomeTable


select  supporter_urn,
        payment_method,
        amount                                                                     as transaction_amount,
        sum(amount) over (partition by supporter_urn)                              as supporter_total,
        sum(amount) over ()                                                        as overall_total,
        100 * amount / sum(amount) over (partition by supporter_urn)               as percentage_1,
        100 * sum(amount) over (partition by supporter_urn) / sum(amount) over ()  as percentage_2,
        100 * amount / sum(amount) over ()                                         as percentage_3,
        row_number() over (partition by supporter_urn order by amount)             as rownum,
        sum(amount) over (partition by payment_method)                             as payment_partition,
        min(amount) over (partition by supporter_urn)                              as supporter_min
from    #IncomeTable
order by supporter_urn


select  supporter_urn,
        payment_method,
        amount,
        row_number() over (order by amount)                        as rownum_1,
        row_number() over (order by payment_method, supporter_urn) as rownum_2,
        rank() over (order by amount)                              as rank_amount,
        rank() over (order by payment_method)                      as rank_payment
from    #IncomeTable



